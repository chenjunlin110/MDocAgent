"use client";

import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { ArrowUp, Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type AgentTurn = {
  role: string;
  text: string;
};

type QAResponse = {
  answer: string;
  agents: AgentTurn[];
  critics: Record<string, string>;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type UploadResponse = {
  id: string;
  filename: string;
  stored_name: string;
  path: string;
  size: number;
  status: "processing" | "ready" | "failed";
  error?: string;
};

type SettingsResponse = {
  api_key_present: boolean;
  base_url: string;
  model: string;
};

type ModelListResponse = {
  models: { id: string }[];
  warning: string;
};

const REGIONS = [
  { label: "us-east-2 (Ohio)", value: "us-east-2" },
  { label: "us-east-1 (N. Virginia)", value: "us-east-1" },
  { label: "us-west-2 (Oregon)", value: "us-west-2" },
  { label: "ap-southeast-3 (Jakarta)", value: "ap-southeast-3" },
  { label: "ap-south-1 (Mumbai)", value: "ap-south-1" },
  { label: "ap-northeast-1 (Tokyo)", value: "ap-northeast-1" },
  { label: "eu-central-1 (Frankfurt)", value: "eu-central-1" },
  { label: "eu-west-1 (Ireland)", value: "eu-west-1" },
  { label: "eu-west-2 (London)", value: "eu-west-2" },
  { label: "eu-south-1 (Milan)", value: "eu-south-1" },
  { label: "eu-north-1 (Stockholm)", value: "eu-north-1" },
  { label: "sa-east-1 (Sao Paulo)", value: "sa-east-1" }
];

const baseUrlForRegion = (region: string) =>
  region ? `https://bedrock-mantle.${region}.api.aws/v1` : "";

const detectRegionFromBaseUrl = (baseUrl: string) => {
  if (!baseUrl) return "";
  const match = baseUrl.match(/bedrock-mantle\.([a-z0-9-]+)\.api\.aws/);
  return match?.[1] ?? "";
};

export default function Home() {
  const [question, setQuestion] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [agentTrace, setAgentTrace] = useState<QAResponse | null>(null);

  const [uploading, setUploading] = useState<boolean>(false);
  const [uploads, setUploads] = useState<UploadResponse[]>([]);
  const [activeUploadId, setActiveUploadId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string>("");
  const [pendingUpload, setPendingUpload] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<string>("");
  const [apiKeySaved, setApiKeySaved] = useState<boolean>(false);
  const [baseUrl, setBaseUrl] = useState<string>("");
  const [region, setRegion] = useState<string>("");
  const [models, setModels] = useState<string[]>([]);
  const [modelsWarning, setModelsWarning] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [loadingModels, setLoadingModels] = useState<boolean>(false);
  const [savingSettings, setSavingSettings] = useState<boolean>(false);
  const [settingsError, setSettingsError] = useState<string>("");
  const [settingsNotice, setSettingsNotice] = useState<string>("");

  const endRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const refreshUploads = async (preferredId?: string) => {
    try {
      const res = await fetch(`${API_URL}/uploads`);
      if (!res.ok) {
        return;
      }
      const payload = (await res.json()) as { uploads: UploadResponse[] };
      const items = payload.uploads ?? [];
      setUploads(items);
      if (preferredId) {
        setActiveUploadId(preferredId);
      } else if (!activeUploadId && items.length) {
        const readyItem = items.find((item) => item.status === "ready");
        setActiveUploadId((readyItem ?? items[0]).id);
      }
    } catch {
      return;
    }
  };

  const loadSettings = async () => {
    try {
      const res = await fetch(`${API_URL}/settings`);
      if (!res.ok) {
        return;
      }
      const payload = (await res.json()) as SettingsResponse;
      setApiKeySaved(payload.api_key_present);
      setBaseUrl(payload.base_url || "");
      setSelectedModel(payload.model || "");
      const detectedRegion = detectRegionFromBaseUrl(payload.base_url || "");
      setRegion(detectedRegion);
    } catch {
      return;
    }
  };

  const buildSettingsPayload = (includeModel: boolean) => {
    const payload: Record<string, string> = {};
    if (apiKey.trim()) {
      payload.api_key = apiKey.trim();
    }
    if (baseUrl.trim()) {
      payload.base_url = baseUrl.trim();
    }
    if (includeModel && selectedModel) {
      payload.model = selectedModel;
    }
    return payload;
  };

  const saveSettings = async (includeModel: boolean, notice?: string) => {
    setSavingSettings(true);
    setSettingsError("");
    setSettingsNotice("");
    try {
      const payload = buildSettingsPayload(includeModel);
      const res = await fetch(`${API_URL}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const message = await res.json();
        throw new Error(message.detail || "Failed to save settings.");
      }
      const data = (await res.json()) as SettingsResponse;
      setApiKeySaved(data.api_key_present);
      setBaseUrl(data.base_url || "");
      setSelectedModel(data.model || "");
      if (notice) {
        setSettingsNotice(notice);
      }
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected settings error.";
      setSettingsError(message);
      return false;
    } finally {
      setSavingSettings(false);
    }
  };

  const loadModels = async () => {
    setLoadingModels(true);
    setSettingsError("");
    setModelsWarning("");
    try {
      const res = await fetch(`${API_URL}/bedrock/models`);
      if (!res.ok) {
        const message = await res.json();
        throw new Error(message.detail || "Failed to load models.");
      }
      const data = (await res.json()) as ModelListResponse;
      setModels(data.models.map((item) => item.id));
      setModelsWarning(data.warning || "");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected model error.";
      setSettingsError(message);
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    refreshUploads();
    loadSettings();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [question]);

  const activeUpload = uploads.find((item) => item.id === activeUploadId) ?? null;
  const maxQuestionChars = 1200;
  const usagePercent = Math.min(100, Math.round((question.length / maxQuestionChars) * 100));

  const handleReset = () => {
    setMessages([]);
    setAgentTrace(null);
    setError("");
    setQuestion("");
  };

  const handleQuestionChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setQuestion(event.target.value);
    event.currentTarget.style.height = "0px";
    event.currentTarget.style.height = `${event.currentTarget.scrollHeight}px`;
  };

  const handleRegionChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const nextRegion = event.target.value;
    setRegion(nextRegion);
    setBaseUrl(baseUrlForRegion(nextRegion));
  };

  const handleModelFocus = () => {
    if (!models.length && !loadingModels) {
      void loadModels();
    }
  };

  const handleSend = async () => {
    if (!activeUploadId) {
      setError("Upload a PDF first.");
      return;
    }
    if (activeUpload && activeUpload.status !== "ready") {
      setError("Selected PDF is still processing.");
      return;
    }
    if (!question.trim()) {
      setError("Type a question to continue.");
      return;
    }

    const userMessage = question.trim();
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setQuestion("");
    setError("");
    setLoading(true);
    setAgentTrace(null);

    try {
      const res = await fetch(`${API_URL}/qa`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          upload_id: activeUploadId,
          question: userMessage,
          top_k: 4
        })
      });

      if (!res.ok) {
        const payload = await res.json();
        throw new Error(payload.detail || "Request failed.");
      }

      const data = (await res.json()) as QAResponse;
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
      setAgentTrace(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const uploadPdf = async (file: File) => {
    const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
    if (!isPdf) {
      setUploadError("Only PDF files are supported.");
      return;
    }
    setUploadError("");
    setUploading(true);
    setPendingUpload(file.name);
    let createdId: string | undefined;
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_URL}/uploads`, {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        const payload = await res.json();
        throw new Error(payload.detail || "Upload failed.");
      }
      const data = (await res.json()) as UploadResponse;
      setUploads((prev) => [data, ...prev]);
      setActiveUploadId(data.id);
      createdId = data.id;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error.";
      setUploadError(message);
    } finally {
      setUploading(false);
      setPendingUpload(null);
      await refreshUploads(createdId);
    }
  };

  const handleUploadChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || uploading) return;
    await uploadPdf(file);
  };

  const handleDeleteUpload = async (uploadId: string) => {
    if (uploadId === activeUploadId) {
      setActiveUploadId(null);
    }
    try {
      const res = await fetch(`${API_URL}/uploads/${uploadId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const payload = (await res.json()) as { detail?: string };
        throw new Error(payload.detail || "Delete failed.");
      }
    } catch (error) {
      console.error(error);
    } finally {
      await refreshUploads();
    }
  };

  return (
    <main className="flex min-h-screen">
      <aside className="flex w-full max-w-xs flex-col gap-6 border-r border-border bg-sidebar/95 px-6 py-8 text-sidebar-foreground">
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-sidebar-foreground/60">
            MDocAgent
          </p>
          <h1 className="text-2xl font-semibold">Workspace</h1>
        </div>

        <button
          type="button"
          onClick={handleReset}
          className="rounded-lg border border-sidebar-foreground/20 bg-sidebar-foreground/5 px-4 py-2 text-left text-sm font-semibold text-sidebar-foreground/90 transition hover:bg-sidebar-foreground/10"
        >
          + New chat
        </button>

        <div className="space-y-4">
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-sidebar-foreground/60">Recent uploads</p>
            {uploads.length ? (
              <div className="space-y-2 text-xs text-sidebar-foreground/70">
                {uploads.slice(0, 4).map((item) => {
                  const isActive = item.id === activeUploadId;
                  const statusTone =
                    item.status === "ready"
                      ? "bg-emerald-400"
                      : item.status === "failed"
                      ? "bg-red-400"
                      : "bg-amber-300";
                  return (
                    <div
                      key={item.id}
                      className={`flex items-stretch overflow-hidden rounded-md border transition ${
                        isActive
                          ? "border-accent bg-sidebar-foreground/15"
                          : "border-sidebar-foreground/10 bg-sidebar-foreground/5 hover:bg-sidebar-foreground/10"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => setActiveUploadId(item.id)}
                        className="flex-1 p-2 text-left"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className={`h-2.5 w-2.5 rounded-full ${statusTone}`} />
                            <p className="font-semibold text-sidebar-foreground/90">{item.filename}</p>
                          </div>
                          {item.status === "ready" ? null : (
                            <span
                              className={`text-[10px] uppercase tracking-[0.2em] ${
                                item.status === "failed" ? "text-red-200" : "text-sidebar-foreground/60"
                              }`}
                            >
                              {item.status}
                            </span>
                          )}
                        </div>
                        <p className="text-[11px] text-sidebar-foreground/70">
                          {Math.round(item.size / 1024)} KB
                        </p>
                        {item.error ? (
                          <p className="mt-1 text-[11px] text-red-200">{item.error}</p>
                        ) : null}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDeleteUpload(item.id)}
                        className="flex w-10 items-center justify-center border-l border-sidebar-foreground/10 text-sidebar-foreground/70 transition hover:bg-sidebar-foreground/10"
                        aria-label={`Delete ${item.filename}`}
                        title={`Delete ${item.filename}`}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-sidebar-foreground/60">No uploads yet.</p>
            )}
          </div>
          <div className="space-y-3 rounded-lg border border-sidebar-foreground/10 bg-sidebar-foreground/5 p-3">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-sidebar-foreground/60">
                Bedrock Mantle
              </p>
              {apiKeySaved ? (
                <span className="rounded-full border border-emerald-200/60 bg-emerald-100/60 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-800">
                  key saved
                </span>
              ) : null}
            </div>
            <Input
              type="password"
              placeholder="API key"
              value={apiKey}
              onChange={(event) => {
                setApiKey(event.target.value);
                setSettingsNotice("");
              }}
            />
            <Select value={region} onChange={handleRegionChange}>
              <option value="">Select region</option>
              {REGIONS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </Select>
            <Select
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
              onMouseDown={handleModelFocus}
              onFocus={handleModelFocus}
            >
              <option value="">
                {loadingModels ? "Loading models..." : models.length ? "Select model" : "Open to load models"}
              </option>
              {models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </Select>
            <Button
              type="button"
              onClick={() => {
                if (!region) {
                  setSettingsError("Select a region before saving the model.");
                  return;
                }
                void saveSettings(true, "Settings saved.");
              }}
              disabled={savingSettings || !selectedModel || !region}
              className="h-9 w-full rounded-md bg-primary text-xs font-semibold text-white"
            >
              {savingSettings ? "Saving..." : "Use model"}
            </Button>
            {modelsWarning ? (
              <p className="text-[11px] text-amber-200">{modelsWarning}</p>
            ) : null}
            {settingsNotice ? (
              <p className="text-[11px] text-emerald-200">{settingsNotice}</p>
            ) : null}
            {settingsError ? <p className="text-[11px] text-red-200">{settingsError}</p> : null}
          </div>
          <div className="rounded-lg border border-sidebar-foreground/10 bg-sidebar-foreground/5 px-3 py-2 text-xs text-sidebar-foreground/70">
            Upload PDFs to build a local document RAG.
          </div>
        </div>

      </aside>

      <section className="flex min-h-screen flex-1 flex-col">
        <header className="flex flex-wrap items-center justify-between gap-4 border-b border-border bg-panel/80 px-8 py-4 backdrop-blur">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground">Session</p>
            <p className="text-xs text-muted-foreground">
              {activeUpload ? `Active PDF: ${activeUpload.filename}` : "No PDF selected"}
            </p>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-8 py-10">
          {messages.length === 0 ? (
            <div className="mx-auto max-w-2xl space-y-6 text-center">
              <h3 className="text-3xl font-semibold text-foreground">Ask your document a question</h3>
            </div>
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-6">
              {messages.map((message, index) => {
                const isUser = message.role === "user";
                return (
                  <div key={`${message.role}-${index}`} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm shadow-sm ${
                        isUser
                          ? "bg-gradient-to-br from-primary via-primary to-primary/90 text-white ring-1 ring-primary/30"
                          : "border border-border/60 bg-white/80 text-foreground shadow-[0_6px_18px_-12px_rgba(15,23,42,0.6)]"
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                );
              })}
              {loading ? (
                <div className="flex justify-start">
                  <div className="flex items-center gap-2 rounded-2xl border border-border bg-white/80 px-4 py-3 text-sm text-muted-foreground shadow-sm">
                    <span className="h-2 w-2 animate-bounce rounded-full bg-primary/70" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-primary/50 [animation-delay:150ms]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-primary/40 [animation-delay:300ms]" />
                  </div>
                </div>
              ) : null}

              {agentTrace ? (
                <details className="rounded-2xl border border-border bg-panel p-4 text-sm text-muted-foreground">
                  <summary className="cursor-pointer text-sm font-semibold text-foreground">Agent trace</summary>
                  <div className="mt-3 space-y-4">
                    <div className="grid gap-3 sm:grid-cols-3">
                      {agentTrace.agents.map((agent) => (
                        <div key={agent.role} className="rounded-xl border border-border bg-white/70 p-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            {agent.role}
                          </p>
                          <p className="mt-2 whitespace-pre-wrap text-xs text-foreground/80">{agent.text}</p>
                        </div>
                      ))}
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div className="rounded-xl border border-border bg-white/70 p-3">
                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Text critic</p>
                        <p className="mt-2 whitespace-pre-wrap text-xs text-foreground/80">
                          {agentTrace.critics?.text || "-"}
                        </p>
                      </div>
                      <div className="rounded-xl border border-border bg-white/70 p-3">
                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Image critic</p>
                        <p className="mt-2 whitespace-pre-wrap text-xs text-foreground/80">
                          {agentTrace.critics?.image || "-"}
                        </p>
                      </div>
                    </div>
                  </div>
                </details>
              ) : null}
            </div>
          )}
          <div ref={endRef} />
        </div>

        <div className="border-t border-border bg-panel/90 px-8 py-5 shadow-[0_-15px_45px_-25px_rgba(15,23,42,0.35)]">
          <div className="mx-auto max-w-3xl">
            <div className="float-in space-y-3">
              <div className="rounded-xl border border-border/60 bg-white px-3 py-2">
                <input
                  id="upload-input"
                  type="file"
                  accept="application/pdf"
                  className="hidden"
                  disabled={uploading}
                  onChange={handleUploadChange}
                />
                <div className="relative">
                  <Textarea
                    ref={textareaRef}
                    value={question}
                    onChange={handleQuestionChange}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        handleSend();
                      }
                    }}
                    placeholder="Ask, search, or chat..."
                    rows={1}
                    className="min-h-[24px] resize-none overflow-hidden rounded-none border-0 bg-transparent px-0 py-0 pr-20 pb-9 text-base leading-[1.2] shadow-none focus-visible:ring-0"
                  />
                  <div className="absolute bottom-0 left-0 flex items-center">
                    <label
                      htmlFor="upload-input"
                      title="Add PDF"
                      className={`flex h-9 w-9 items-center justify-center rounded-full border border-border/60 bg-panel/80 text-muted-foreground transition hover:border-primary/40 hover:text-foreground ${
                        uploading ? "cursor-not-allowed opacity-60" : "cursor-pointer"
                      }`}
                    >
                      <Plus className="h-4 w-4" aria-hidden="true" />
                      <span className="sr-only">Add PDF</span>
                    </label>
                  </div>
                  <div className="absolute bottom-0 right-0 flex items-center">
                    <Button
                      onClick={handleSend}
                      disabled={loading}
                      className="h-9 w-9 rounded-full bg-primary p-0 text-white shadow-sm hover:brightness-110"
                      aria-label="Send message"
                    >
                      {loading ? (
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                      ) : (
                        <ArrowUp className="h-4 w-4" strokeWidth={2.2} aria-hidden="true" />
                      )}
                    </Button>
                  </div>
                </div>
                <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
                  {pendingUpload ? (
                    <span className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-white/80 px-3 py-1">
                      <span className="h-3 w-3 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-primary" />
                      Uploading {pendingUpload}...
                    </span>
                  ) : (
                    <span>{question ? `${question.length} chars` : "Waiting for your question."}</span>
                  )}
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
                {uploads.length ? (
                  <div className="flex flex-wrap items-center gap-2">
                    {uploads.slice(0, 3).map((item) => (
                      <span key={item.id} className="rounded-full border border-border/60 bg-white/80 px-3 py-1">
                        {item.filename}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p>No uploads yet.</p>
                )}
              </div>
              {uploadError ? (
                <p className="mt-2 rounded-xl border border-red-200/70 bg-red-50/70 px-3 py-2 text-xs text-red-700">
                  {uploadError}
                </p>
              ) : null}
              {error ? (
                <p className="mt-2 rounded-xl border border-red-200/70 bg-red-50/70 px-3 py-2 text-xs text-red-700">
                  {error}
                </p>
              ) : null}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
