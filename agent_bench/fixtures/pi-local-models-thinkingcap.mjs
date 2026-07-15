// Pi / Oh-my-pi extension: ThinkingCap via OpenAI shim (:8091).
// Copy to $HOME/.pi/local-models.mjs or set extensions[] in agent/settings.json.
export default function (pi) {
  pi.registerProvider("local-ai", {
    apiKey: process.env.OPENAI_API_KEY || "local",
    baseUrl: process.env.OPENAI_BASE_URL || "http://127.0.0.1:8091/v1",
    api: "openai-completions",
    models: [
      {
        id: "thinkingcap",
        name: "ThinkingCap via shim",
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 65536,
        maxTokens: 8192,
        compat: { supportsDeveloperRole: false, supportsReasoningEffort: false },
      },
    ],
  });

  pi.registerProvider("local", {
    apiKey: "local",
    baseUrl: process.env.ANTHROPIC_BASE_URL || "http://127.0.0.1:8080",
    api: "anthropic-messages",
    models: [
      {
        id: "thinkingcap",
        name: "ThinkingCap via Kevlar",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 65536,
        maxTokens: 8192,
      },
    ],
  });
}
