"use client";

import { useMemo, useState, useRef, useEffect } from "react";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type Conversation = {
  id: string;
  title: string;
  messages: ChatMessage[];
};

type GenerateResponse = {
  text: string;
  latency_ms: number;
  model_params_m: number;
  device: string;
  session_id?: string;
};

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const [showToolbar, setShowToolbar] = useState(true);
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const [copiedSystemPrompt, setCopiedSystemPrompt] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const [maxNewTokens, setMaxNewTokens] = useState(150);
  const [temperature, setTemperature] = useState(0.6);
  const [topK, setTopK] = useState(20);
  const [topP, setTopP] = useState(0.85);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [useHistory, setUseHistory] = useState(true);
  const [maxHistoryMessages, setMaxHistoryMessages] = useState(8);
  const [systemPrompt, setSystemPrompt] = useState("你是一个有帮助的助手。请用中文回答所有问题。");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const initialized = useRef(false);

  const apiBase = useMemo(() => {
    const fromEnv = process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "");
    if (fromEnv) return fromEnv;
    if (typeof window !== "undefined") return window.location.origin;
    return "";
  }, []);

  const endpoint = `${apiBase}/v1/generate`;

  const currentConv = conversations.find((c) => c.id === currentConvId);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      createNewConversation();
    }
  }, []);

  const createNewConversation = () => {
    const id = Date.now().toString();
    const newConv: Conversation = {
      id,
      title: "新对话",
      messages: [],
    };
    setConversations((prev) => [newConv, ...prev]);
    setCurrentConvId(id);
    setMessages([]);
    setSessionId(null);
    setError(null);
  };

  const switchConversation = (id: string) => {
    setCurrentConvId(id);
    const conv = conversations.find((c) => c.id === id);
    if (conv) {
      setMessages(conv.messages);
      setError(null);
    }
  };

  const deleteConversation = (id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (currentConvId === id) {
      const remaining = conversations.filter((c) => c.id !== id);
      if (remaining.length > 0) {
        switchConversation(remaining[0].id);
      } else {
        createNewConversation();
      }
    }
  };

  const updateConversationTitle = (id: string, title: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title } : c))
    );
  };

  const copyToClipboard = (text: string, idx: number) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx(null), 2000);
    });
  };

  const copySystemPromptToClipboard = () => {
    navigator.clipboard.writeText(systemPrompt).then(() => {
      setCopiedSystemPrompt(true);
      setTimeout(() => setCopiedSystemPrompt(false), 2000);
    });
  };

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  const removeSelectedFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const onFilesSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    setSelectedFiles((prev) => [...prev, ...files]);
    e.target.value = "";
  };

  const handlePasteImage = (e: React.ClipboardEvent<HTMLInputElement>) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      if (items[i].kind === "file" && items[i].type.startsWith("image/")) {
        e.preventDefault();
        const file = items[i].getAsFile();
        if (file) {
          setSelectedFiles((prev) => [...prev, file]);
        }
      }
    }
  };

  const sendMessage = async () => {
    const prompt = input.trim();
    if ((!prompt && selectedFiles.length === 0) || loading || !currentConvId) return;

    setError(null);
    setLoading(true);
    setInput("");

    // 如果只有图片没有文字，使用默认提示
    const userPrompt = prompt || "请描述这张图片";
    const finalPrompt = systemPrompt ? `${systemPrompt}\n${userPrompt}` : userPrompt;
    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: prompt || "[图片]" }];
    setMessages(nextMessages);

    try {
      // 如果有文件，使用multipart端点；否则使用JSON端点
      let usedEndpoint = endpoint;
      let fetchOptions: RequestInit = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: finalPrompt,
          max_new_tokens: maxNewTokens,
          temperature,
          top_k: topK,
          top_p: topP,
          repetition_penalty: repetitionPenalty,
          session_id: sessionId,
          use_history: useHistory,
          max_history_messages: maxHistoryMessages,
        }),
      };

      // 如果有选中的文件，改用multipart端点和FormData
      if (selectedFiles.length > 0) {
        usedEndpoint = `${apiBase}/v1/generate-multipart`;
        const formData = new FormData();
        formData.append("prompt", finalPrompt);
        formData.append("max_new_tokens", maxNewTokens.toString());
        formData.append("temperature", temperature.toString());
        formData.append("top_k", topK.toString());
        formData.append("top_p", topP.toString());
        formData.append("repetition_penalty", repetitionPenalty.toString());
        if (sessionId) formData.append("session_id", sessionId);
        formData.append("use_history", useHistory.toString());
        formData.append("max_history_messages", maxHistoryMessages.toString());
        
        // 只添加第一个图片文件（目前只支持单张图片）
        if (selectedFiles[0]) {
          formData.append("image", selectedFiles[0]);
        }

        fetchOptions = {
          method: "POST",
          body: formData,
        };
        // 不要设置Content-Type，让浏览器自动设置multipart boundary
      }

      console.log("Sending request to:", usedEndpoint);
      try {
        const response = await fetch(usedEndpoint, fetchOptions);
        console.log("Response received:", response.status, response.statusText);

        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || `Request failed: ${response.status}`);
        }

        const text = await response.text();
        console.log("Response text:", text.substring(0, 200));
        const data = JSON.parse(text) as GenerateResponse;
        console.log("Parsed data:", data);
        if (data.session_id) {
          setSessionId(data.session_id);
        }

        const updatedMessages: ChatMessage[] = [
          ...nextMessages,
          { role: "assistant", content: data.text },
        ];
        setMessages(updatedMessages);
        setSelectedFiles([]);

        // Update conversation title from first user message
        if (nextMessages.length === 1) {
          const title = prompt.length > 30 ? prompt.substring(0, 30) + "..." : prompt;
          updateConversationTitle(currentConvId, title);
        }

        // Save to conversation
        setConversations((prev) =>
          prev.map((c) =>
            c.id === currentConvId ? { ...c, messages: updatedMessages } : c
          )
        );
      } catch (fetchError) {
        console.error("Fetch error details:", fetchError);
        throw fetchError;
      }
    } catch (err) {
      console.error("Request error:", err);
      setError(err instanceof Error ? err.message : "请求失败");
      setMessages(nextMessages);
    } finally {
      setLoading(false);
    }
  };

  if (!currentConvId) {
    return null;
  }

  return (
    <div className="flex h-screen bg-white">
      {/* 左侧边栏 */}
      {showSidebar && (
        <aside className="w-64 border-r border-zinc-200 flex flex-col bg-zinc-50">
          <div className="p-4 border-b border-zinc-200">
            <button
              onClick={createNewConversation}
              className="w-full rounded-lg border border-zinc-300 px-4 py-2 text-sm hover:bg-zinc-100"
            >
              + 新建对话
            </button>
          </div>
          <div className="flex-1 overflow-y-auto">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`border-l-2 px-4 py-3 cursor-pointer transition ${
                  currentConvId === conv.id
                    ? "border-l-zinc-900 bg-white"
                    : "border-l-transparent hover:bg-zinc-100"
                }`}
                onClick={() => switchConversation(conv.id)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 truncate">
                    <p className="text-sm font-medium text-zinc-900 truncate">
                      {conv.title}
                    </p>
                    <p className="text-xs text-zinc-500 mt-1">
                      {conv.messages.length} 条消息
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteConversation(conv.id);
                    }}
                    className="text-zinc-400 hover:text-red-500"
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
          <div className="border-t border-zinc-200 p-4">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="w-full rounded-lg px-4 py-2 text-sm text-zinc-600 hover:bg-zinc-100"
            >
              ⚙️ 设置
            </button>
          </div>
        </aside>
      )}

      {/* 主聊天区域 */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* 顶部工具栏 */}
        {showToolbar && (
        <div className="border-b border-zinc-200 px-6 py-4 flex items-center justify-between">
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="text-zinc-600 hover:text-zinc-900"
          >
            {showSidebar ? "←" : "→"}
          </button>
          <div />
          <div className="flex items-center gap-4">
            <div className="text-xs text-zinc-500">
              {currentConv?.messages.length || 0} 条消息
            </div>
            <button
              onClick={() => setShowToolbar(false)}
              className="text-xs text-zinc-400 hover:text-zinc-600"
              title="隐藏工具栏"
            >
              ✕
            </button>
          </div>
        </div>
        )}
        {!showToolbar && (
        <div className="px-4 py-2">
          <button
            onClick={() => setShowToolbar(true)}
            className="text-xs text-zinc-400 hover:text-zinc-600"
            title="显示工具栏"
          >
            ▼ 工具栏
          </button>
        </div>
        )}

        {/* 消息区域 */}
        <div className="flex-1 overflow-y-auto px-6 py-8">
          <div className="mx-auto max-w-4xl space-y-6">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center text-center">
              <div>
                <p className="text-3xl font-bold text-zinc-900">开始对话</p>
                <p className="mt-2 text-zinc-500">输入你的问题或想法</p>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={`${msg.role}-${idx}`}
                className={`flex w-full gap-2 group ${msg.role === "assistant" ? "justify-start" : "justify-end"}`}
              >
                {msg.role === "assistant" && (
                  <button
                    onClick={() => copyToClipboard(msg.content, idx)}
                    className="self-start mt-2 p-2 rounded opacity-0 group-hover:opacity-100 transition-opacity hover:bg-zinc-200 flex-shrink-0"
                    title="复制"
                  >
                    {copiedIdx === idx ? (
                      <span className="text-xs text-green-600">✓ 已复制</span>
                    ) : (
                      <svg className="w-4 h-4 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    )}
                  </button>
                )}
                <div className={`relative group/msg ${msg.role === "assistant" ? "flex-1" : "max-w-xl"}`}>
                  <div
                    className={`rounded-2xl px-6 py-4 whitespace-pre-wrap leading-7 ${
                      msg.role === "assistant"
                        ? "bg-zinc-100 text-zinc-900"
                        : "bg-zinc-900 text-white"
                    }`}
                  >
                    {msg.content}
                  </div>
                  {msg.role === "user" && (
                    <button
                      onClick={() => copyToClipboard(msg.content, idx)}
                      className="absolute -right-10 top-1/2 -translate-y-1/2 p-2 rounded opacity-0 group-hover/msg:opacity-100 transition-opacity hover:bg-zinc-800"
                      title="复制"
                    >
                      {copiedIdx === idx ? (
                        <span className="text-xs text-green-600">✓ 已复制</span>
                      ) : (
                        <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                      )}
                    </button>
                  )}
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="flex w-full justify-start">
              <div className="bg-zinc-100 text-zinc-900 rounded-2xl px-6 py-4 flex items-center gap-2">
                <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: "0.4s" }}></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
          </div>
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="px-6 py-4 bg-red-50 border-b border-red-200">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* 输入区域 */}
        <div className="border-t border-zinc-200 px-6 py-6">
          <div className="mx-auto max-w-4xl">
            <div className="relative">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*,audio/*,.pdf,.txt,.md,.json,.csv,.doc,.docx"
                className="hidden"
                onChange={onFilesSelected}
              />

              {selectedFiles.length > 0 && (
                <div className="mb-2 flex flex-wrap gap-2 rounded-xl border border-zinc-200 bg-zinc-50 p-2">
                  {selectedFiles.map((file, index) => (
                    <div
                      key={`${file.name}-${index}`}
                      className="inline-flex items-center gap-2 rounded-full bg-white border border-zinc-200 px-3 py-1 text-xs text-zinc-700"
                    >
                      <span className="max-w-[180px] truncate" title={file.name}>{file.name}</span>
                      <button
                        type="button"
                        onClick={() => removeSelectedFile(index)}
                        className="text-zinc-500 hover:text-red-500"
                        title="移除附件"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !loading) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                placeholder="输入提示词... (Shift+Enter 换行)"
                className="w-full rounded-2xl border border-zinc-300 pl-14 pr-24 py-4 text-base outline-none focus:border-zinc-500 focus:ring-2 focus:ring-zinc-200 resize-none"
                rows={2}
              />
              <button
                type="button"
                onClick={openFilePicker}
                className="absolute left-3 bottom-3 flex h-9 w-9 items-center justify-center rounded-full border border-zinc-300 bg-white text-2xl leading-none text-zinc-700 hover:bg-zinc-50"
                title="添加图片/文件/音频"
              >
                +
              </button>
              <button
                onClick={sendMessage}
                disabled={loading || (!input.trim() && selectedFiles.length === 0)}
                className="absolute right-2 bottom-2 rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:bg-zinc-300"
              >
                发送
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 右侧设置面板 */}
      {showSettings && (
        <aside className="w-72 border-l border-zinc-200 bg-zinc-50 p-6 overflow-y-auto flex flex-col gap-6">
          <div>
            <h2 className="text-lg font-semibold text-zinc-900 mb-4">参数设置</h2>
            <div className="space-y-4">
              {/* 系统提示词 */}
              <div className="border-b border-zinc-300 pb-4">
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-xs font-medium text-zinc-700">
                    系统提示词
                  </label>
                  <button
                    onClick={copySystemPromptToClipboard}
                    className="p-1 rounded hover:bg-zinc-200 transition-colors"
                    title="复制提示词"
                  >
                    {copiedSystemPrompt ? (
                      <span className="text-xs text-green-600">✓</span>
                    ) : (
                      <svg className="w-3 h-3 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    )}
                  </button>
                </div>
                <input
                  type="text"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  onPaste={handlePasteImage}
                  className="w-full px-3 py-2 text-sm border border-zinc-700 rounded-lg outline-none focus:border-zinc-600 focus:ring-2 focus:ring-zinc-800 bg-zinc-900 text-white placeholder-zinc-400"
                  placeholder="输入系统提示词..."
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-700 mb-2">
                  max_new_tokens: {maxNewTokens}
                </label>
                <input
                  type="range"
                  min={1}
                  max={1024}
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-zinc-700 mb-2">
                  temperature: {temperature.toFixed(2)}
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={2}
                  step={0.05}
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-zinc-700 mb-2">
                  top_k: {topK}
                </label>
                <input
                  type="range"
                  min={1}
                  max={1000}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-zinc-700 mb-2">
                  top_p: {topP.toFixed(2)}
                </label>
                <input
                  type="range"
                  min={0.01}
                  max={1}
                  step={0.01}
                  value={topP}
                  onChange={(e) => setTopP(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-zinc-700 mb-2">
                  repetition_penalty: {repetitionPenalty.toFixed(2)}
                </label>
                <input
                  type="range"
                  min={1}
                  max={2}
                  step={0.05}
                  value={repetitionPenalty}
                  onChange={(e) => setRepetitionPenalty(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          <div className="border-t border-zinc-300 pt-4">
            <div className="flex items-center justify-between mb-3">
              <label className="text-xs font-medium text-zinc-700">
                使用历史上下文
              </label>
              <input
                type="checkbox"
                checked={useHistory}
                onChange={(e) => setUseHistory(e.target.checked)}
                className="h-4 w-4"
              />
            </div>
            {useHistory && (
              <div>
                <label className="block text-xs font-medium text-zinc-700 mb-2">
                  历史消息数: {maxHistoryMessages}
                </label>
                <input
                  type="range"
                  min={0}
                  max={50}
                  value={maxHistoryMessages}
                  onChange={(e) => setMaxHistoryMessages(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            )}
          </div>

          <div className="text-xs text-zinc-500">
            <p>API Base: {apiBase}</p>
            {sessionId && <p className="mt-1">Session: {sessionId.slice(0, 8)}...</p>}
          </div>
        </aside>
      )}
    </div>
  );
}
