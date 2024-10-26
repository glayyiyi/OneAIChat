import { getServerSideConfig } from "../config/server";
import { ModelProvider } from "../constant";
import { prettyObject } from "../utils/format";
import { NextRequest, NextResponse } from "next/server";
import { auth } from "./auth";
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
  InvokeModelCommandInput,
} from "@aws-sdk/client-bedrock-runtime";

const ALLOWED_PATH = new Set(["invoke"]);

export async function handle(
  req: NextRequest,
  { params }: { params: { path: string[] } },
) {
  console.log("[Bedrock Route] params ", params);

  if (req.method === "OPTIONS") {
    return NextResponse.json({ body: "OK" }, { status: 200 });
  }

  const subpath = params.path.join("/");

  if (!ALLOWED_PATH.has(subpath)) {
    console.log("[Bedrock Route] forbidden path ", subpath);
    return NextResponse.json(
      {
        error: true,
        msg: "you are not allowed to request " + subpath,
      },
      {
        status: 403,
      },
    );
  }

  const authResult = auth(req, ModelProvider.Bedrock);
  if (authResult.error) {
    return NextResponse.json(authResult, {
      status: 401,
    });
  }

  try {
    const response = await request(req);
    return response;
  } catch (e) {
    console.error("[Bedrock] ", e);
    return NextResponse.json(prettyObject(e));
  }
}

const serverConfig = getServerSideConfig();

function formatRequestBody(modelId: string, messages: any[]) {
  if (modelId.startsWith("anthropic.claude")) {
    // Format for Claude models
    const roleMap: Record<string, string> = {
      system: "system",
      user: "user",
      assistant: "assistant",
    };

    // Clean up messages and ensure proper role mapping
    const cleanMessages = messages.map((m: any) => ({
      role: roleMap[m.role] || m.role,
      content: m.content.trim(),
    }));

    // Ensure messages alternate between user and assistant
    const formattedMessages = cleanMessages.reduce(
      (acc: any[], msg: any, i: number) => {
        if (i === 0) {
          // First message can be from user
          return [msg];
        }
        const prevMsg = acc[acc.length - 1];
        if (prevMsg.role === msg.role && msg.role === "user") {
          // If we have consecutive user messages, insert a placeholder assistant message
          acc.push({
            role: "assistant",
            content: "I understand. Please continue.",
          });
        }
        acc.push(msg);
        return acc;
      },
      [],
    );

    // For Claude 3 and 3.5 models
    if (modelId.includes("claude-3") || modelId.includes("claude-3-5")) {
      return {
        anthropic_version: "bedrock-2023-05-31",
        messages: formattedMessages,
        max_tokens: 2048,
        temperature: 0.7,
        top_p: 0.9,
      };
    }
    // For Claude 2 and earlier models
    else {
      const formattedText = formattedMessages
        .map(
          (m: any) =>
            `\n\n${m.role === "user" ? "Human" : "Assistant"}: ${m.content}`,
        )
        .join("");

      return {
        prompt: formattedText + "\n\nAssistant:",
        max_tokens_to_sample: 2048,
        temperature: 0.7,
        top_p: 0.9,
        stop_sequences: ["\n\nHuman:", "\n\nAssistant:"],
        anthropic_version: "bedrock-2023-05-31",
      };
    }
  } else if (modelId.startsWith("amazon.titan")) {
    return {
      inputText: messages[messages.length - 1].content,
      textGenerationConfig: {
        maxTokenCount: 2048,
        temperature: 0.7,
        topP: 0.9,
        stopSequences: [],
      },
    };
  } else if (
    modelId.startsWith("meta.llama") ||
    modelId.includes("inference-llama")
  ) {
    // Format for Llama models (including inference profile ARNs)
    const lastMessage = messages[messages.length - 1];
    return {
      messages: [
        {
          role: "user",
          content: lastMessage.content,
        },
      ],
      temperature: 0.7,
      top_p: 0.9,
      max_tokens: 2048,
    };
  } else if (modelId.startsWith("mistral.")) {
    // For Mistral models, format as a conversation prompt
    // Filter out messages that look like JSON responses
    const validMessages = messages.filter(
      (m: any) =>
        !m.content.includes("error") && !m.content.includes('{"outputs"'),
    );

    // Get the last user message
    const lastUserMessage = validMessages[validMessages.length - 1];
    if (!lastUserMessage) {
      throw new Error("No valid user message found");
    }

    return {
      prompt: `<s>[INST] ${lastUserMessage.content.trim()} [/INST]`,
      max_tokens: 2048,
      temperature: 0.7,
      top_p: 0.9,
    };
  } else {
    throw new Error(`Unsupported model: ${modelId}`);
  }
}

async function parseResponse(modelId: string, response: Response) {
  const text = await response.text();
  try {
    const responseJson = JSON.parse(text);

    if (modelId.startsWith("anthropic.claude")) {
      if (modelId.includes("claude-3") || modelId.includes("claude-3-5")) {
        const content =
          responseJson.content?.[0]?.text ||
          responseJson.content ||
          responseJson.completion;
        return new Response(content);
      } else {
        return new Response(responseJson.completion);
      }
    } else if (modelId.startsWith("amazon.titan")) {
      return new Response(responseJson.results[0].outputText);
    } else if (
      modelId.startsWith("meta.llama") ||
      modelId.includes("inference-llama")
    ) {
      const content =
        responseJson.messages?.[0]?.content ||
        responseJson.generation ||
        responseJson.text ||
        responseJson.output;
      return new Response(content);
    } else if (modelId.startsWith("mistral.")) {
      // Extract the actual text content from Mistral's response
      if (
        responseJson.outputs &&
        responseJson.outputs[0] &&
        responseJson.outputs[0].text
      ) {
        return new Response(responseJson.outputs[0].text.trim());
      }
      return new Response(text);
    } else {
      throw new Error(`Unsupported model: ${modelId}`);
    }
  } catch (e) {
    console.error("[Bedrock] Failed to parse response JSON:", e);
    return new Response(text);
  }
}

async function request(req: NextRequest) {
  const controller = new AbortController();

  const region = req.headers.get("X-Region") || "us-east-1";
  const accessKeyId = req.headers.get("X-Access-Key") || "";
  const secretAccessKey = req.headers.get("X-Secret-Key") || "";
  const sessionToken = req.headers.get("X-Session-Token");
  const inferenceProfile = req.headers.get("X-Inference-Profile");

  if (!accessKeyId || !secretAccessKey) {
    return NextResponse.json(
      {
        error: true,
        message: "Missing AWS credentials",
      },
      {
        status: 401,
      },
    );
  }

  console.log("[Bedrock] Using region:", region);

  const client = new BedrockRuntimeClient({
    region,
    credentials: {
      accessKeyId,
      secretAccessKey,
      sessionToken: sessionToken || undefined,
    },
  });

  const timeoutId = setTimeout(
    () => {
      controller.abort();
    },
    10 * 60 * 1000,
  );

  try {
    const body = await req.json();
    const { messages, model } = body;

    console.log("[Bedrock] Invoking model:", model);
    console.log("[Bedrock] Messages:", messages);

    // For Llama models, we need an inference profile
    if (model.startsWith("meta.llama")) {
      if (!inferenceProfile) {
        throw new Error(
          "Llama models require an inference profile ARN. Please follow these steps:\n" +
            "1. Go to AWS Bedrock console\n" +
            "2. Create an inference profile for the Llama model\n" +
            "3. Copy the inference profile ARN\n" +
            "4. Add it to your application's AWS settings under 'Inference Profile ARN'",
        );
      }
      if (!inferenceProfile.includes("inference")) {
        throw new Error(
          "Invalid inference profile ARN. The ARN should contain 'inference' and be in the format: \n" +
            "arn:aws:bedrock:region:account:inference-profile/profile-name",
        );
      }
    }

    const requestBody = formatRequestBody(model, messages);
    const jsonString = JSON.stringify(requestBody);
    const input: InvokeModelCommandInput = {
      modelId: model.startsWith("meta.llama") ? inferenceProfile : model,
      contentType: "application/json",
      accept: "application/json",
      body: Uint8Array.from(Buffer.from(jsonString)),
    };

    console.log("[Bedrock] Request input:", {
      ...input,
      body: requestBody,
    });

    const command = new InvokeModelCommand(input);
    const response = await client.send(command);

    console.log("[Bedrock] Got response");

    // Create a Response object from the raw response
    const responseBody = new TextDecoder().decode(response.body);
    const result = await parseResponse(model, new Response(responseBody));

    console.log("[Bedrock] Parsed response:", await result.clone().text());

    return result;
  } catch (e) {
    console.error("[Bedrock] Request error:", e);
    return NextResponse.json(
      {
        error: true,
        message: e instanceof Error ? e.message : "Unknown error",
      },
      {
        status: 500,
      },
    );
  } finally {
    clearTimeout(timeoutId);
  }
}
