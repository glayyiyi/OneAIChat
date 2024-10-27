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

// Helper function to get the base model type from modelId
function getModelType(modelId: string): string {
  // If it's an inference profile ARN, extract the model name
  if (modelId.includes("inference-profile")) {
    const match = modelId.match(/us\.(meta\.llama.+?)$/);
    if (match) return match[1];
  }
  return modelId;
}

// Helper function to clean error messages from chat history
function cleanErrorMessages(messages: any[]): any[] {
  return messages.filter(
    (m) =>
      !m.content.includes('"error": true') &&
      !m.content.includes("ValidationException") &&
      !m.content.includes("Unsupported model"),
  );
}

// Helper function to clean repeated content
function cleanRepeatedContent(text: string): string {
  // Split by common separators
  const parts = text.split(/\n+|(?:A:|Assistant:)\s*/);
  // Take the first non-empty part
  return parts.find((p) => p.trim().length > 0) || text;
}

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
  // Get the base model type
  const baseModel = getModelType(modelId);

  // Clean error messages from chat history
  const cleanMessages = cleanErrorMessages(messages);

  if (baseModel.startsWith("anthropic.claude")) {
    // Format for Claude models
    const roleMap: Record<string, string> = {
      system: "system",
      user: "user",
      assistant: "assistant",
    };

    // Clean up messages and ensure proper role mapping
    const formattedMessages = cleanMessages.map((m: any) => ({
      role: roleMap[m.role] || m.role,
      content: m.content.trim(),
    }));

    // Ensure messages alternate between user and assistant
    const finalMessages = formattedMessages.reduce(
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
    if (baseModel.includes("claude-3") || baseModel.includes("claude-3-5")) {
      return {
        anthropic_version: "bedrock-2023-05-31",
        messages: finalMessages,
        max_tokens: 2048,
        temperature: 0.7,
        top_p: 0.9,
      };
    }
    // For Claude 2 and earlier models
    else {
      const formattedText = finalMessages
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
  } else if (baseModel.startsWith("amazon.titan")) {
    return {
      inputText: cleanMessages[cleanMessages.length - 1].content,
      textGenerationConfig: {
        maxTokenCount: 2048,
        temperature: 0.7,
        topP: 0.9,
        stopSequences: [],
      },
    };
  } else if (baseModel.startsWith("meta.llama")) {
    // Format for Llama models
    const formattedMessages = cleanMessages
      .map(
        (m: any) =>
          `${m.role === "user" ? "Human" : "Assistant"}: ${m.content.trim()}`,
      )
      .join("\n\n");

    // Check if using inference profile
    if (modelId.includes("inference-profile")) {
      const systemPrompt =
        "You are a helpful AI assistant. Provide direct and concise responses. Do not repeat yourself or generate additional dialogue.";
      return {
        prompt: `${systemPrompt}\n\n${formattedMessages}\n\nAssistant:`,
        temperature: 0.6,
        top_p: 0.9,
      };
    } else {
      // For base model
      return {
        inputs: [cleanMessages],
        parameters: {
          max_new_tokens: 512,
          temperature: 0.6,
          top_p: 0.9,
        },
      };
    }
  } else if (baseModel.startsWith("mistral.")) {
    // For Mistral models, format as a conversation prompt
    // Get the last user message
    const lastUserMessage = cleanMessages[cleanMessages.length - 1];
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

    // Get the base model type for response parsing
    const baseModel = getModelType(modelId);

    if (baseModel.startsWith("anthropic.claude")) {
      if (baseModel.includes("claude-3") || baseModel.includes("claude-3-5")) {
        const content =
          responseJson.content?.[0]?.text ||
          responseJson.content ||
          responseJson.completion;
        return new Response(content);
      } else {
        return new Response(responseJson.completion);
      }
    } else if (baseModel.startsWith("amazon.titan")) {
      return new Response(responseJson.results[0].outputText);
    } else if (baseModel.startsWith("meta.llama")) {
      const content =
        responseJson.generation || responseJson.completion || text;
      // Clean up repeated content
      const cleanContent = cleanRepeatedContent(content);
      return new Response(cleanContent);
    } else if (baseModel.startsWith("mistral.")) {
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

    const requestBody = formatRequestBody(model, messages);
    const jsonString = JSON.stringify(requestBody);
    const input: InvokeModelCommandInput = {
      modelId: model,
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
