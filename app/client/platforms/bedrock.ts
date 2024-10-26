import { ApiPath } from "../../constant";
import { ChatOptions, getHeaders, LLMApi, SpeechOptions } from "../api";
import {
  useAccessStore,
  useAppConfig,
  useChatStore,
  usePluginStore,
} from "../../store";
import { getMessageTextContent } from "../../utils";
import { stream } from "../../utils/chat";
import Locale from "../../locales";

export interface BedrockChatRequest {
  model: string;
  messages: Array<{
    role: string;
    content: string;
  }>;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
}

export class BedrockApi implements LLMApi {
  speech(options: SpeechOptions): Promise<ArrayBuffer> {
    throw new Error("Method not implemented.");
  }

  extractMessage(res: any) {
    console.log("[Response] bedrock response: ", res);
    return res;
  }

  async chat(options: ChatOptions): Promise<void> {
    const shouldStream = !!options.config.stream;

    const modelConfig = {
      ...useAppConfig.getState().modelConfig,
      ...useChatStore.getState().currentSession().mask.modelConfig,
      ...{
        model: options.config.model,
      },
    };

    const accessStore = useAccessStore.getState();

    if (
      !accessStore.awsRegion ||
      !accessStore.awsAccessKeyId ||
      !accessStore.awsSecretAccessKey
    ) {
      console.log("AWS credentials are not set");
      let responseText = "";
      const responseTexts = [responseText];
      responseTexts.push(Locale.Error.Unauthorized);
      responseText = responseTexts.join("\n\n");
      options.onFinish(responseText);
      return;
    }

    // Validate inference profile for Llama models
    if (
      modelConfig.model.startsWith("meta.llama") &&
      !accessStore.awsInferenceProfile
    ) {
      console.error("Inference profile is required for Llama models");
      options.onError?.(
        new Error(
          "Inference profile ARN is required for Llama models. Please configure an inference profile in your AWS Bedrock settings.",
        ),
      );
      return;
    }

    const messages = options.messages.map((v) => ({
      role: v.role,
      content: getMessageTextContent(v),
    }));

    const requestBody: BedrockChatRequest = {
      messages,
      stream: shouldStream,
      model: modelConfig.model,
      max_tokens: modelConfig.max_tokens,
      temperature: modelConfig.temperature,
      top_p: modelConfig.top_p,
    };

    console.log("[Bedrock] Request:", {
      model: modelConfig.model,
      messages: messages,
    });

    const controller = new AbortController();
    options.onController?.(controller);

    const headers: Record<string, string> = {
      ...getHeaders(),
      "X-Region": accessStore.awsRegion,
      "X-Access-Key": accessStore.awsAccessKeyId,
      "X-Secret-Key": accessStore.awsSecretAccessKey,
    };

    if (accessStore.awsSessionToken) {
      headers["X-Session-Token"] = accessStore.awsSessionToken;
    }

    // Add inference profile for Llama models
    if (modelConfig.model.startsWith("meta.llama")) {
      headers["X-Inference-Profile"] = accessStore.awsInferenceProfile;
      console.log(
        "[Bedrock] Using inference profile:",
        accessStore.awsInferenceProfile,
      );
    }

    try {
      if (shouldStream) {
        let responseText = "";
        const pluginStore = usePluginStore.getState();
        const currentSession = useChatStore.getState().currentSession();
        const [tools, funcs] = pluginStore.getAsTools(
          currentSession.mask?.plugin || [],
        );

        await stream(
          `${ApiPath.Bedrock}/invoke`,
          requestBody,
          headers,
          Array.isArray(tools) ? tools : [],
          funcs || {},
          controller,
          (chunk: string) => {
            try {
              responseText += chunk;
              return chunk;
            } catch (e) {
              console.error("[Request] parse error", chunk, e);
              return "";
            }
          },
          (
            requestPayload: any,
            toolCallMessage: any,
            toolCallResult: any[],
          ) => {
            console.log("[Bedrock] processToolMessage", {
              requestPayload,
              toolCallMessage,
              toolCallResult,
            });
          },
          options,
        );
      } else {
        const response = await fetch(`${ApiPath.Bedrock}/invoke`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        if (!response.ok) {
          const error = await response.text();
          console.error("[Bedrock] Error response:", error);
          throw new Error(`Bedrock API error: ${error}`);
        }

        const text = await response.text();
        options.onFinish(text);
      }
    } catch (e) {
      console.error("[Bedrock] Chat error:", e);
      options.onError?.(e as Error);
    }
  }

  async usage() {
    return {
      used: 0,
      total: 0,
    };
  }

  async models() {
    return [];
  }
}
