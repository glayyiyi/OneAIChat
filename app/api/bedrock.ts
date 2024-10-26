import { getServerSideConfig } from "../config/server";
import { ModelProvider } from "../constant";
import { prettyObject } from "../utils/format";
import { NextRequest, NextResponse } from "next/server";
import { auth } from "./auth";
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
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

    // For Claude 3 and 3.5 models
    if (modelId.includes("claude-3") || modelId.includes("claude-3-5")) {
      return {
        anthropic_version: "bedrock-2023-05-31",
        messages: cleanMessages,
        max_tokens: 2048,
        temperature: 0.7,
        top_p: 0.9,
      };
    }
    // For Claude 2 and earlier models
    else {
      const formattedMessages = cleanMessages
        .map(
          (m: any) =>
            `\n\n${m.role === "user" ? "Human" : "Assistant"}: ${m.content}`,
        )
        .join("");

      return {
        prompt: formattedMessages + "\n\nAssistant:",
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
  } else if (modelId.startsWith("meta.llama")) {
    // Format for Llama models
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
    // Format for Mistral models
    const cleanMessages = messages.map((m: any) => ({
      role: m.role,
      content: m.content.trim(),
    }));

    return {
      messages: cleanMessages,
      max_tokens: 2048,
      temperature: 0.7,
      top_p: 0.9,
    };
  } else {
    throw new Error(`Unsupported model: ${modelId}`);
  }
}

function parseResponse(modelId: string, responseBody: string) {
  const responseJson = JSON.parse(responseBody);

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
  } else if (modelId.startsWith("meta.llama")) {
    const text =
      responseJson.messages?.[0]?.content ||
      responseJson.generation ||
      responseJson.text ||
      responseJson.output;
    return new Response(text);
  } else if (modelId.startsWith("mistral.")) {
    const text =
      responseJson.messages?.[0]?.content ||
      responseJson.outputs?.[0]?.text ||
      responseJson.outputs?.[0]?.content;
    return new Response(text);
  } else {
    throw new Error(`Unsupported model: ${modelId}`);
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

    const requestBody = formatRequestBody(model, messages);

    const input = {
      modelId: model,
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify(requestBody),
    };

    // Add inference profile if provided
    if (
      inferenceProfile &&
      (model.startsWith("meta.llama") || model.startsWith("mistral."))
    ) {
      (input as any).inferenceProfile = inferenceProfile;
    }

    console.log("[Bedrock] Request input:", {
      ...input,
      body: JSON.parse(input.body),
    });

    const command = new InvokeModelCommand(input);
    const response = await client.send(command);

    console.log("[Bedrock] Got response");

    // Parse response
    const responseBody = new TextDecoder().decode(response.body);
    const result = parseResponse(model, responseBody);

    console.log("[Bedrock] Response:", result);

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
