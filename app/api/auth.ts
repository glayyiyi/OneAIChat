import { NextRequest } from "next/server";
import { getServerSideConfig } from "../config/server";
import md5 from "spark-md5";
import { ACCESS_CODE_PREFIX, ModelProvider } from "../constant";

function getIP(req: NextRequest) {
  let ip = req.ip ?? req.headers.get("x-real-ip");
  const forwardedFor = req.headers.get("x-forwarded-for");

  if (!ip && forwardedFor) {
    ip = forwardedFor.split(",").at(0) ?? "";
  }

  return ip;
}

function parseApiKey(bearToken: string) {
  const token = bearToken.trim().replaceAll("Bearer ", "").trim();
  const isApiKey = !token.startsWith(ACCESS_CODE_PREFIX);

  return {
    accessCode: isApiKey ? "" : token.slice(ACCESS_CODE_PREFIX.length),
    apiKey: isApiKey ? token : "",
  };
}

export function auth(req: NextRequest, modelProvider: ModelProvider) {
  const authToken = req.headers.get("Authorization") ?? "";

  // check if it is openai api key or user token
  const { accessCode, apiKey } = parseApiKey(authToken);

  const hashedCode = md5.hash(accessCode ?? "").trim();

  const serverConfig = getServerSideConfig();
  console.log("[Auth] allowed hashed codes: ", [...serverConfig.codes]);
  console.log("[Auth] got access code:", accessCode);
  console.log("[Auth] hashed access code:", hashedCode);
  console.log("[User IP] ", getIP(req));
  console.log("[Time] ", new Date().toLocaleString());

  if (serverConfig.needCode && !serverConfig.codes.has(hashedCode) && !apiKey) {
    return {
      error: true,
      msg: !accessCode ? "empty access code" : "wrong access code",
    };
  }

  if (serverConfig.hideUserApiKey && !!apiKey) {
    return {
      error: true,
      msg: "you are not allowed to access with your own api key",
    };
  }

  // Special handling for Bedrock
  if (modelProvider === ModelProvider.Bedrock) {
    const region = req.headers.get("X-Region");
    const accessKeyId = req.headers.get("X-Access-Key");
    const secretKey = req.headers.get("X-Secret-Key");

    console.log("[Auth] Bedrock credentials:", {
      region,
      accessKeyId: accessKeyId ? "***" : undefined,
      secretKey: secretKey ? "***" : undefined,
    });

    // Check if AWS credentials are provided
    if (!region || !accessKeyId || !secretKey) {
      return {
        error: true,
        msg: "Missing AWS credentials. Please configure Region, Access Key ID, and Secret Access Key in settings.",
      };
    }

    return { error: false };
  }

  // if user does not provide an api key, inject system api key
  if (!apiKey) {
    const serverConfig = getServerSideConfig();

    let systemApiKey: string | undefined;
    console.log("[Auth] modelProvider:" + modelProvider);
    switch (modelProvider) {
      case ModelProvider.Stability:
        systemApiKey = serverConfig.stabilityApiKey;
        break;
      case ModelProvider.GeminiPro:
        systemApiKey = serverConfig.googleApiKey;
        break;
      case ModelProvider.Claude:
        systemApiKey = serverConfig.anthropicApiKey;
        break;
      case ModelProvider.Doubao:
        systemApiKey = serverConfig.bytedanceApiKey;
        break;
      case ModelProvider.Ernie:
        systemApiKey = serverConfig.baiduApiKey;
        break;
      case ModelProvider.Qwen:
        systemApiKey = serverConfig.alibabaApiKey;
        break;
      case ModelProvider.Moonshot:
        systemApiKey = serverConfig.moonshotApiKey;
        break;
      case ModelProvider.Iflytek:
        systemApiKey =
          serverConfig.iflytekApiKey + ":" + serverConfig.iflytekApiSecret;
        break;
      case ModelProvider.XAI:
        systemApiKey = serverConfig.xaiApiKey;
        break;
      case ModelProvider.ChatGLM:
        systemApiKey = serverConfig.chatglmApiKey;
        break;
      case ModelProvider.GPT:
      default:
        if (req.nextUrl.pathname.includes("azure/deployments")) {
          systemApiKey = serverConfig.azureApiKey;
        } else {
          systemApiKey = serverConfig.apiKey;
        }
    }

    if (systemApiKey) {
      console.log("[Auth] use system api key");
      req.headers.set("Authorization", `Bearer ${systemApiKey}`);
    } else {
      console.log("[Auth] admin did not provide an api key......");
    }
  } else {
    console.log("[Auth] use user api key");
  }

  return {
    error: false,
  };
}
