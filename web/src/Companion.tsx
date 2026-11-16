import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'

type Article = { url: string; title: string; paragraphs: string[] }
type Source = {
  id: number
  kind: string
  title: string
  url: string
  paragraphIndex: number
  text: string
}
type TurnStatus = 'streaming' | 'done' | 'stopped' | 'error'
type Turn = {
  id: number
  question: string
  answer: string
  sources: Source[]
  status: TurnStatus
  error?: string
  metrics?: Record<string, number | string | boolean>
}
export type VoiceQuestion = { text: string; id: number; releasedAt: number }

type Chat = { id: string; title: string; updatedAt: number; turns: Turn[] }

const newChatId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

// Changes whenever a chat has something new worth sending to the server.
const chatSignature = (chat: Chat) => {
  const last = chat.turns[chat.turns.length - 1]
  return `${chat.updatedAt}:${chat.turns.length}:${last?.answer.length ?? 0}:${last?.status ?? ''}`
}

function mergeChats(local: Chat[], remote: Chat[]): Chat[] {
  const merged = new Map<string, Chat>()
  for (const chat of [...local, ...remote]) {
    const existing = merged.get(chat.id)
    if (!existing || chat.updatedAt >= existing.updatedAt)
      merged.set(chat.id, chat)
  }
  return [...merged.values()]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_CHATS_PER_ARTICLE)
}

const MAX_HISTORY_TURNS = 16
const CHATS_STORAGE_KEY = 'margin:companion-chats'
const MAX_CHATS_PER_ARTICLE = 30

function readAllChats(): Record<string, Chat[]> {
  try {
    const parsed: unknown = JSON.parse(
      window.localStorage.getItem(CHATS_STORAGE_KEY) ?? '{}',
    )
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, Chat[]>)
      : {}
  } catch {
    return {}
  }
}

function loadChats(articleUrl: string): Chat[] {
  const chats = readAllChats()[articleUrl]
  if (!Array.isArray(chats)) return []
  return chats
    .filter((c) => c && Array.isArray(c.turns) && c.turns.length)
    .map((c) => ({
      ...c,
      id: String(c.id),
      // An answer that was mid-stream when the page closed is incomplete.
      turns: c.turns.map((turn) =>
        turn.status === 'streaming'
          ? { ...turn, status: 'stopped' as const }
          : turn,
      ),
    }))
    .sort((a, b) => b.updatedAt - a.updatedAt)
}

function saveChats(articleUrl: string, chats: Chat[]) {
  try {
    const all = readAllChats()
    if (chats.length) all[articleUrl] = chats
    else delete all[articleUrl]
    window.localStorage.setItem(CHATS_STORAGE_KEY, JSON.stringify(all))
  } catch {
    // Storage full or blocked: the conversation still works for this session.
  }
}

function formatChatDate(timestamp: number) {
  const date = new Date(timestamp)
  const now = new Date()
  if (date.toDateString() === now.toDateString())
    return new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: '2-digit',
    }).format(date)
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
  }).format(date)
}

function HistoryIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <path
        d="M2.5 8a5.5 5.5 0 1 0 1.6-3.9M2.5 2.5v2.6h2.6M8 5v3.2l2 1.3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function NewChatIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <path
        d="M8 3v10M3 8h10"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

const SUGGESTIONS: { text: string; scope?: 'passage' | 'library' }[] = [
  { text: 'What is the author really arguing here?' },
  { text: 'Explain this passage in plain words.' },
  { text: 'Where else have I read about this idea?', scope: 'library' },
]

/* ---------- Lightweight markdown for answers ---------- */

const INLINE_PATTERN =
  /(\*\*[^*]+\*\*|`[^`]+`|\[\d+\]|\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g

function renderInline(
  text: string,
  sources: Source[],
  onNavigate: (source: Source) => void,
): ReactNode[] {
  return text.split(INLINE_PATTERN).map((part, n) => {
    if (!part) return null
    const citation = part.match(/^\[(\d+)\]$/)
    if (citation) {
      const source = sources.find((s) => s.id === Number(citation[1]))
      return source ? (
        <button
          type="button"
          key={n}
          className="companion-citation"
          title={`${source.kind === 'thought' ? 'Your thought' : source.title}`}
          aria-label={`Source ${source.id}: ${source.title}`}
          onClick={() => onNavigate(source)}
        >
          {source.id}
        </button>
      ) : (
        <Fragment key={n}>{part}</Fragment>
      )
    }
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4)
      return <strong key={n}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2)
      return <code key={n}>{part.slice(1, -1)}</code>
    if (
      (part.startsWith('*') && part.endsWith('*')) ||
      (part.startsWith('_') && part.endsWith('_'))
    )
      return <em key={n}>{part.slice(1, -1)}</em>
    return <Fragment key={n}>{part}</Fragment>
  })
}

function renderAnswer(
  text: string,
  sources: Source[],
  onNavigate: (source: Source) => void,
): ReactNode[] {
  const lines = text.replace(/\r/g, '').split('\n')
  const blocks: ReactNode[] = []
  let paragraph: string[] = []
  let list: { ordered: boolean; items: string[] } | null = null

  const inline = (value: string) => renderInline(value, sources, onNavigate)
  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(<p key={blocks.length}>{inline(paragraph.join(' '))}</p>)
      paragraph = []
    }
  }
  const flushList = () => {
    if (list) {
      const items = list.items.map((item, i) => <li key={i}>{inline(item)}</li>)
      blocks.push(
        list.ordered ? (
          <ol key={blocks.length}>{items}</ol>
        ) : (
          <ul key={blocks.length}>{items}</ul>
        ),
      )
      list = null
    }
  }

  for (const raw of lines) {
    const line = raw.trim()
    const bullet = line.match(/^[-*•]\s+(.*)$/)
    const numbered = line.match(/^\d+[.)]\s+(.*)$/)
    const heading = line.match(/^#{1,6}\s+(.*)$/)
    const quote = line.match(/^>\s?(.*)$/)

    if (!line) {
      flushParagraph()
      flushList()
    } else if (bullet || numbered) {
      flushParagraph()
      const ordered = Boolean(numbered)
      if (list && list.ordered !== ordered) flushList()
      if (!list) list = { ordered, items: [] }
      list.items.push((bullet ?? numbered)![1])
    } else if (heading) {
      flushParagraph()
      flushList()
      blocks.push(
        <p key={blocks.length} className="companion-answer-heading">
          {inline(heading[1].replace(/\*\*/g, ''))}
        </p>,
      )
    } else if (quote) {
      flushParagraph()
      flushList()
      blocks.push(
        <blockquote key={blocks.length}>{inline(quote[1])}</blockquote>,
      )
    } else if (list && /^\s{2,}/.test(raw)) {
      list.items[list.items.length - 1] += ` ${line}`
    } else {
      flushList()
      paragraph.push(line)
    }
  }
  flushParagraph()
  flushList()
  return blocks
}

/* ---------- Component ---------- */

export function Companion({
  apiUrl,
  apiKey,
  clientId,
  article,
  library,
  paragraphIndex,
  passageLabel,
  hidden,
  voiceQuestion,
  voiceDraft,
  onNavigate,
}: {
  apiUrl: string
  apiKey: string
  clientId: string
  article: Article
  library: Article[]
  paragraphIndex: number
  passageLabel: string | null
  hidden?: boolean
  voiceQuestion: VoiceQuestion | null
  voiceDraft: string
  onNavigate: (source: Source) => void
}) {
  const [question, setQuestion] = useState('')
  const [scope, setScope] = useState<'passage' | 'library'>('passage')
  const [chats, setChats] = useState<Chat[]>(() => loadChats(article.url))
  const [chatId, setChatId] = useState<string>(
    () => chats[0]?.id ?? newChatId(),
  )
  const syncedRef = useRef(new Map<string, string>())
  const [turns, setTurns] = useState<Turn[]>(
    () => chats.find((c) => c.id === chatId)?.turns ?? [],
  )
  const [showHistory, setShowHistory] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [indexStatus, setIndexStatus] = useState('')
  const [showJump, setShowJump] = useState(false)
  const controller = useRef<AbortController | null>(null)
  const lastVoice = useRef(0)
  const turnsRef = useRef(turns)
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const pendingScrollId = useRef<number | null>(null)

  useEffect(() => {
    turnsRef.current = turns
  }, [turns])
  useEffect(() => () => controller.current?.abort(), [])

  // Save the open conversation into this article's history.
  useEffect(() => {
    if (!turns.length) return
    const timeout = window.setTimeout(() => {
      setChats((previous) => {
        const chat: Chat = {
          id: chatId,
          title: turns[0].question,
          updatedAt:
            previous.find((c) => c.id === chatId)?.turns.length ===
              turns.length && !busy
              ? (previous.find((c) => c.id === chatId)?.updatedAt ?? Date.now())
              : Date.now(),
          turns: turns.map(({ metrics: _metrics, ...turn }) => turn),
        }
        const next = [chat, ...previous.filter((c) => c.id !== chatId)]
          .sort((a, b) => b.updatedAt - a.updatedAt)
          .slice(0, MAX_CHATS_PER_ARTICLE)
        saveChats(article.url, next)
        return next
      })
    }, 300)
    return () => window.clearTimeout(timeout)
  }, [turns, chatId, busy, article.url])

  // Load this article's chats from the server and merge with the local cache.
  useEffect(() => {
    const abort = new AbortController()
    void (async () => {
      try {
        const params = new URLSearchParams({
          clientId,
          articleUrl: article.url,
        })
        const response = await fetch(`${apiUrl}/companion/chats?${params}`, {
          signal: abort.signal,
        })
        if (!response.ok) return
        const { chats: remote } = (await response.json()) as { chats: Chat[] }
        for (const chat of remote)
          syncedRef.current.set(chat.id, chatSignature(chat))
        setChats((local) => {
          const merged = mergeChats(local, remote)
          saveChats(article.url, merged)
          return merged
        })
        // Nothing asked yet in this session: open the most recent chat.
        if (!turnsRef.current.length) {
          const latest = mergeChats(loadChats(article.url), remote)[0]
          if (latest) {
            setChatId(latest.id)
            turnsRef.current = latest.turns
            setTurns(latest.turns)
          }
        }
      } catch {
        // Offline or database down: the local cache keeps history working.
      }
    })()
    return () => abort.abort()
  }, [apiUrl, clientId, article.url])

  // Push finished changes to the server (not every streamed token).
  useEffect(() => {
    if (busy) return
    for (const chat of chats) {
      const signature = chatSignature(chat)
      if (syncedRef.current.get(chat.id) === signature) continue
      syncedRef.current.set(chat.id, signature)
      void fetch(`${apiUrl}/companion/chats/${encodeURIComponent(chat.id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          clientId,
          articleUrl: article.url,
          title: chat.title.slice(0, 300),
          turns: chat.turns,
          updatedAt: new Date(chat.updatedAt).toISOString(),
        }),
      })
        .then((response) => {
          if (!response.ok) syncedRef.current.delete(chat.id)
        })
        .catch(() => syncedRef.current.delete(chat.id))
    }
  }, [chats, busy, apiUrl, clientId, article.url])

  useEffect(() => {
    if (!apiKey) return
    const abort = new AbortController()
    const articles = [
      ...new Map([...library, article].map((a) => [a.url, a])).values(),
    ]
    let failed = 0
    void (async () => {
      for (let i = 0; i < articles.length; i++) {
        setIndexStatus(
          `Preparing your library · ${i + 1} of ${articles.length}`,
        )
        try {
          const response = await fetch(`${apiUrl}/companion/index`, {
            method: 'POST',
            signal: abort.signal,
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${apiKey}`,
            },
            body: JSON.stringify({ ...articles[i], clientId }),
          })
          if (!response.ok) failed++
        } catch {
          if (abort.signal.aborted) return
          failed++
        }
      }
      setIndexStatus(
        failed
          ? `${failed} of ${articles.length} articles couldn’t be indexed — library answers may be incomplete.`
          : `Searching across ${articles.length} ${articles.length === 1 ? 'article' : 'articles'}`,
      )
    })()
    return () => abort.abort()
  }, [apiKey, apiUrl, clientId, article, library])

  // Bring each newly asked question to the top of the conversation.
  useLayoutEffect(() => {
    const id = pendingScrollId.current
    const container = scrollRef.current
    if (id === null || !container) return
    const el = container.querySelector<HTMLElement>(`[data-turn-id="${id}"]`)
    if (!el) return
    pendingScrollId.current = null
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    container.scrollTo({
      top: Math.max(0, el.offsetTop - 4),
      behavior: reduce ? 'auto' : 'smooth',
    })
  }, [turns])

  const updateJump = useCallback(() => {
    const container = scrollRef.current
    const last = container?.querySelector<HTMLElement>(
      '.companion-turn:last-child',
    )
    if (!container || !last) return setShowJump(false)
    const top = container.scrollTop
    const bottom = top + container.clientHeight
    setShowJump(
      last.offsetTop > bottom - 48 || last.offsetTop + last.offsetHeight < top,
    )
  }, [])
  useEffect(updateJump, [turns, updateJump])

  const jumpToLatest = () => {
    const container = scrollRef.current
    const last = container?.querySelector<HTMLElement>(
      '.companion-turn:last-child',
    )
    if (container && last)
      container.scrollTo({ top: last.offsetTop - 4, behavior: 'smooth' })
  }

  const markInterrupted = (previous: Turn[]) =>
    previous.map((t) =>
      t.status === 'streaming' ? { ...t, status: 'stopped' as const } : t,
    )

  const ask = useCallback(
    async (
      text: string,
      options: { releasedAt?: number; scope?: 'passage' | 'library' } = {},
    ) => {
      const trimmed = text.trim()
      if (!trimmed) return
      if (!apiKey) {
        setError('Add your NVIDIA key in Settings to ask Margin.')
        return
      }
      const activeScope = options.scope ?? scope
      controller.current?.abort()
      const abort = new AbortController()
      controller.current = abort
      const started = performance.now()
      let firstAt: number | null = null
      // Recent turns go verbatim; older questions travel as a short recap.
      const answered = turnsRef.current.filter((t) => t.answer)
      const history = answered.slice(-MAX_HISTORY_TURNS).flatMap((t) => [
        { role: 'user', content: t.question },
        {
          role: 'assistant',
          // Old citation numbers point at old passages; drop them.
          content: t.answer.replace(/\s?\[\d+\]/g, '').slice(0, 6000),
        },
      ])
      const earlierQuestions = answered
        .slice(0, -MAX_HISTORY_TURNS)
        .slice(-30)
        .map((t) => t.question.slice(0, 300))
      const turnId = Date.now() + Math.random()
      pendingScrollId.current = turnId
      setBusy(true)
      setError('')
      setQuestion('')
      setShowHistory(false)
      setTurns((previous) => [
        ...markInterrupted(previous),
        {
          id: turnId,
          question: trimmed,
          answer: '',
          sources: [],
          status: 'streaming',
        },
      ])
      const update = (fn: (turn: Turn) => Turn) => {
        setTurns((previous) =>
          previous.map((t) => (t.id === turnId ? fn(t) : t)),
        )
      }
      const live = () => controller.current === abort && !abort.signal.aborted
      try {
        const start = Math.max(0, paragraphIndex - 2)
        const response = await fetch(`${apiUrl}/companion/ask`, {
          method: 'POST',
          signal: abort.signal,
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            clientId,
            question: trimmed,
            articleUrl: article.url,
            title: article.title,
            paragraphIndex,
            contextStart: start,
            context: article.paragraphs.slice(start, paragraphIndex + 3),
            scope: activeScope,
            history,
            earlierQuestions,
          }),
        })
        if (!response.ok || !response.body)
          throw new Error(
            'Margin couldn’t start an answer. Check your key in Settings.',
          )
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let completed = false
        const consume = (line: string) => {
          if (!line.trim() || !live()) return
          const event = JSON.parse(line)
          if (event.type === 'error') throw new Error(event.detail)
          if (event.type === 'sources')
            update((t) => ({ ...t, sources: event.sources }))
          if (event.type === 'delta') {
            if (firstAt === null) firstAt = performance.now()
            update((t) => ({ ...t, answer: t.answer + event.text }))
          }
          if (event.type === 'done') {
            completed = true
            update((t) => ({
              ...t,
              status: 'done',
              metrics: {
                ...event.metrics,
                clientFirstAnswerMs: Math.round(
                  (firstAt ?? performance.now()) - started,
                ),
                ...(options.releasedAt
                  ? {
                      releaseToTranscriptMs: Math.round(
                        started - options.releasedAt,
                      ),
                      releaseToFirstAnswerMs: Math.round(
                        (firstAt ?? performance.now()) - options.releasedAt,
                      ),
                    }
                  : {}),
              },
            }))
          }
        }
        while (true) {
          const { value, done } = await reader.read()
          buffer += decoder.decode(value, { stream: !done })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''
          lines.forEach(consume)
          if (done) break
        }
        consume(buffer)
        if (!completed && live())
          throw new Error('The answer was cut off partway.')
      } catch (e) {
        if (!abort.signal.aborted)
          update((t) => ({
            ...t,
            status: 'error',
            error: e instanceof Error ? e.message : 'Margin couldn’t answer.',
          }))
      } finally {
        if (controller.current === abort) setBusy(false)
      }
    },
    [apiKey, apiUrl, article, clientId, paragraphIndex, scope],
  )

  useEffect(() => {
    if (!voiceQuestion || voiceQuestion.id === lastVoice.current) return
    lastVoice.current = voiceQuestion.id
    void ask(voiceQuestion.text, { releasedAt: voiceQuestion.releasedAt })
  }, [voiceQuestion, ask])

  const stop = () => {
    controller.current?.abort()
    controller.current = null
    setTurns(markInterrupted)
    setBusy(false)
  }

  const retry = (turn: Turn) => {
    const remaining = turnsRef.current.filter((t) => t.id !== turn.id)
    turnsRef.current = remaining
    setTurns(remaining)
    void ask(turn.question)
  }

  const startNewChat = () => {
    stop()
    setChatId(newChatId())
    setTurns([])
    setError('')
    setShowHistory(false)
    inputRef.current?.focus()
  }

  const openChat = (chat: Chat) => {
    stop()
    setChatId(chat.id)
    turnsRef.current = chat.turns
    setTurns(chat.turns)
    setError('')
    setShowHistory(false)
    pendingScrollId.current = chat.turns[chat.turns.length - 1]?.id ?? null
  }

  const deleteChat = (chat: Chat) => {
    const next = chats.filter((c) => c.id !== chat.id)
    setChats(next)
    saveChats(article.url, next)
    syncedRef.current.delete(chat.id)
    void fetch(
      `${apiUrl}/companion/chats/${encodeURIComponent(chat.id)}?clientId=${encodeURIComponent(clientId)}`,
      { method: 'DELETE' },
    ).catch(() => {})
    if (chat.id === chatId) {
      stop()
      setChatId(newChatId())
      setTurns([])
    }
  }

  // Grow the composer with its content, up to a few lines.
  const composerValue = voiceDraft || question
  useLayoutEffect(() => {
    const input = inputRef.current
    if (!input) return
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, 132)}px`
  }, [composerValue])

  const canSend = Boolean(question.trim()) && !voiceDraft
  const lastIndex = turns.length - 1

  return (
    <section
      className="companion-panel"
      aria-label="Ask Margin"
      hidden={hidden}
    >
      <header className="companion-header">
        <div className="companion-scope" role="group" aria-label="Answer from">
          <button
            type="button"
            aria-pressed={scope === 'passage'}
            onClick={() => setScope('passage')}
          >
            This passage
          </button>
          <button
            type="button"
            aria-pressed={scope === 'library'}
            onClick={() => setScope('library')}
          >
            My library
          </button>
        </div>
        <button
          type="button"
          className="companion-icon-button"
          aria-pressed={showHistory}
          aria-label="Chat history"
          title="Chat history"
          onClick={() => setShowHistory((open) => !open)}
        >
          <HistoryIcon />
          {chats.length > 0 && (
            <span className="companion-icon-count">{chats.length}</span>
          )}
        </button>
        <button
          type="button"
          className="companion-icon-button"
          aria-label="New chat"
          title="New chat"
          disabled={!turns.length && !showHistory}
          onClick={startNewChat}
        >
          <NewChatIcon />
        </button>
      </header>
      <p className="companion-context" role="status">
        {scope === 'library' ? (
          indexStatus || 'Answers will draw on everything you’ve saved.'
        ) : passageLabel ? (
          <>
            Focused on <strong>{passageLabel}</strong>
          </>
        ) : (
          'Tap a passage in the article to focus your question.'
        )}
      </p>

      {showHistory && (
        <div className="companion-history" aria-label="Chat history">
          <div className="companion-history-heading">
            <span>Chats about this article</span>
            <button type="button" onClick={() => setShowHistory(false)}>
              Close
            </button>
          </div>
          {chats.length === 0 ? (
            <p className="companion-history-empty">
              No saved chats yet. Your conversations will appear here.
            </p>
          ) : (
            <ul>
              {chats.map((chat) => (
                <li
                  key={chat.id}
                  className={chat.id === chatId ? 'is-current' : undefined}
                >
                  <button
                    type="button"
                    className="companion-history-open"
                    onClick={() => openChat(chat)}
                  >
                    <span className="companion-history-title">
                      {chat.title}
                    </span>
                    <span className="companion-history-meta">
                      {formatChatDate(chat.updatedAt)} · {chat.turns.length}{' '}
                      {chat.turns.length === 1 ? 'question' : 'questions'}
                      {chat.id === chatId ? ' · open' : ''}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="companion-history-delete"
                    aria-label={`Delete chat: ${chat.title}`}
                    title="Delete chat"
                    onClick={() => deleteChat(chat)}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="companion-scroll-wrap" hidden={showHistory}>
        <div
          className="companion-turns"
          ref={scrollRef}
          onScroll={updateJump}
          aria-live="polite"
          aria-busy={busy}
        >
          {turns.length === 0 ? (
            <div className="companion-empty">
              <span className="companion-empty-mark" aria-hidden="true">
                ✳
              </span>
              <p className="companion-empty-title">Think it through.</p>
              <p className="companion-empty-copy">
                Ask about what you’re reading, or follow an idea across your
                library. Answers cite the passages they draw on.
              </p>
              <div className="companion-suggestions">
                {SUGGESTIONS.map((s) => (
                  <button
                    type="button"
                    key={s.text}
                    onClick={() => {
                      if (s.scope) setScope(s.scope)
                      void ask(s.text, { scope: s.scope })
                    }}
                  >
                    {s.text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            turns.map((turn, i) => (
              <article
                className={`companion-turn${i === lastIndex && turns.length > 1 ? ' is-latest' : ''}${turn.status === 'streaming' ? ' is-streaming' : ''}`}
                key={turn.id}
                data-turn-id={turn.id}
              >
                <p className="companion-question">{turn.question}</p>
                <div className="companion-answer">
                  {turn.answer ? (
                    renderAnswer(turn.answer, turn.sources, onNavigate)
                  ) : turn.status === 'streaming' ? (
                    <p className="companion-thinking">
                      <span className="companion-dots" aria-hidden="true">
                        <span />
                        <span />
                        <span />
                      </span>
                      {turn.sources.length
                        ? `Reading ${turn.sources.length} passages…`
                        : 'Finding the evidence…'}
                    </p>
                  ) : null}
                </div>

                {(turn.status === 'stopped' || turn.status === 'error') && (
                  <div
                    className={`companion-turn-note${turn.status === 'error' ? ' has-error' : ''}`}
                    role={turn.status === 'error' ? 'alert' : undefined}
                  >
                    <span>
                      {turn.status === 'error'
                        ? turn.error
                        : turn.answer
                          ? 'Stopped.'
                          : 'Stopped before answering.'}
                    </span>
                    {i === lastIndex && (
                      <button type="button" onClick={() => retry(turn)}>
                        Try again
                      </button>
                    )}
                  </div>
                )}

                {!!turn.sources.length && turn.status !== 'streaming' && (
                  <details className="companion-evidence">
                    <summary>
                      {turn.sources.length}{' '}
                      {turn.sources.length === 1 ? 'source' : 'sources'}
                    </summary>
                    <div className="companion-evidence-list">
                      {turn.sources.map((source) => (
                        <button
                          key={source.id}
                          type="button"
                          onClick={() => onNavigate(source)}
                        >
                          <span className="companion-evidence-id">
                            {source.id}
                          </span>
                          <span className="companion-evidence-body">
                            <strong>
                              {source.kind === 'thought'
                                ? 'Your thought · '
                                : ''}
                              {source.title}
                            </strong>
                            <span>
                              {source.text.length > 160
                                ? `${source.text.slice(0, 160).trimEnd()}…`
                                : source.text}
                            </span>
                          </span>
                        </button>
                      ))}
                    </div>
                  </details>
                )}

                {import.meta.env.DEV && turn.metrics && (
                  <details className="companion-metrics">
                    <summary>Response timings (dev)</summary>
                    <pre>{JSON.stringify(turn.metrics, null, 2)}</pre>
                  </details>
                )}
              </article>
            ))
          )}
        </div>
        {showJump && (
          <button
            type="button"
            className="companion-jump"
            onClick={jumpToLatest}
          >
            Latest <span aria-hidden="true">↓</span>
          </button>
        )}
      </div>

      <form
        className="companion-composer"
        onSubmit={(e) => {
          e.preventDefault()
          if (canSend) void ask(question)
        }}
      >
        <div className={`companion-input${voiceDraft ? ' is-listening' : ''}`}>
          <textarea
            id="companion-question"
            ref={inputRef}
            maxLength={2000}
            value={composerValue}
            readOnly={Boolean(voiceDraft)}
            aria-label={
              voiceDraft ? 'Live question transcription' : 'Ask Margin'
            }
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={
              turns.length
                ? 'Ask a follow-up…'
                : 'What does the author mean here?'
            }
            rows={1}
            onKeyDown={(event) => {
              if (
                event.key === 'Enter' &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault()
                if (canSend) void ask(question)
              } else if (event.key === 'Escape') {
                event.currentTarget.blur()
              }
            }}
          />
          {busy && !canSend ? (
            <button
              type="button"
              className="companion-send is-stop"
              onClick={stop}
              aria-label="Stop answering"
              title="Stop"
            >
              <span aria-hidden="true">■</span>
            </button>
          ) : (
            <button
              type="submit"
              className="companion-send"
              disabled={!canSend}
              aria-label="Ask Margin"
              title="Ask (Enter)"
            >
              <span aria-hidden="true">↑</span>
            </button>
          )}
        </div>
        <p className="companion-hint">
          {voiceDraft
            ? 'Listening…'
            : 'Enter to ask · Shift+Enter for a new line · Esc, then hold Space to speak'}
        </p>
      </form>
      {error && (
        <p role="alert" className="companion-error">
          {error}
        </p>
      )}
    </section>
  )
}
