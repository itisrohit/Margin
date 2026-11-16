import { useCallback, useEffect, useRef, useState } from 'react'

type Article = { url: string; title: string; paragraphs: string[] }
type Source = {
  id: number
  kind: string
  title: string
  url: string
  paragraphIndex: number
  text: string
}
type Turn = {
  question: string
  answer: string
  sources: Source[]
  metrics?: Record<string, number | string | boolean>
}
export type VoiceQuestion = { text: string; id: number; releasedAt: number }

export function Companion({
  apiUrl,
  apiKey,
  clientId,
  article,
  library,
  paragraphIndex,
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
  voiceQuestion: VoiceQuestion | null
  voiceDraft: string
  onNavigate: (source: Source) => void
}) {
  const [question, setQuestion] = useState('')
  const [scope, setScope] = useState<'passage' | 'library'>('passage')
  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [indexStatus, setIndexStatus] = useState('')
  const controller = useRef<AbortController | null>(null)
  const lastVoice = useRef(0)
  const turnsRef = useRef(turns)
  useEffect(() => {
    turnsRef.current = turns
  }, [turns])
  useEffect(() => () => controller.current?.abort(), [])
  useEffect(() => {
    if (!apiKey) return
    const abort = new AbortController()
    const articles = [
      ...new Map([...library, article].map((a) => [a.url, a])).values(),
    ]
    let failed = 0
    void (async () => {
      for (let i = 0; i < articles.length; i++) {
        setIndexStatus(`Preparing library ${i + 1}/${articles.length}`)
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
          ? `${failed} articles could not be indexed. Library answers may be incomplete.`
          : `${articles.length} articles ready for library questions`,
      )
    })()
    return () => abort.abort()
  }, [apiKey, apiUrl, clientId, article, library])
  const ask = useCallback(
    async (text: string, releasedAt?: number) => {
      if (!text.trim()) return
      if (!apiKey) {
        setError('Add your NVIDIA key in Settings first.')
        return
      }
      controller.current?.abort()
      const abort = new AbortController()
      controller.current = abort
      const started = performance.now()
      let firstAt: number | null = null
      const history = turnsRef.current.slice(-4).flatMap((t) => [
        { role: 'user', content: t.question },
        { role: 'assistant', content: t.answer.slice(0, 6000) },
      ])
      setBusy(true)
      setError('')
      setQuestion('')
      setTurns((previous) => [
        ...previous,
        { question: text, answer: '', sources: [] },
      ])
      const update = (fn: (turn: Turn) => Turn) => {
        if (controller.current !== abort || abort.signal.aborted) return
        setTurns((previous) =>
          previous.map((t, i) => (i === previous.length - 1 ? fn(t) : t)),
        )
      }
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
            question: text,
            articleUrl: article.url,
            title: article.title,
            paragraphIndex,
            contextStart: start,
            context: article.paragraphs.slice(start, paragraphIndex + 3),
            scope,
            history,
          }),
        })
        if (!response.ok || !response.body)
          throw new Error(
            'Could not start the answer. Check Settings and retry.',
          )
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let completed = false
        const consume = (line: string) => {
          if (!line.trim()) return
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
              metrics: {
                ...event.metrics,
                clientFirstAnswerMs: Math.round(
                  (firstAt ?? performance.now()) - started,
                ),
                ...(releasedAt
                  ? {
                      releaseToTranscriptMs: Math.round(started - releasedAt),
                      releaseToFirstAnswerMs: Math.round(
                        (firstAt ?? performance.now()) - releasedAt,
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
        if (!completed)
          throw new Error('The answer was interrupted. Please retry.')
      } catch (e) {
        if (!abort.signal.aborted)
          setError(e instanceof Error ? e.message : 'Could not answer.')
      } finally {
        if (controller.current === abort) setBusy(false)
      }
    },
    [apiKey, apiUrl, article, clientId, paragraphIndex, scope],
  )
  useEffect(() => {
    if (!voiceQuestion || voiceQuestion.id === lastVoice.current) return
    lastVoice.current = voiceQuestion.id
    void ask(voiceQuestion.text, voiceQuestion.releasedAt)
  }, [voiceQuestion, ask])
  return (
    <section className="companion-panel" aria-label="Reading companion">
      <div className="companion-heading">
        <span className="eyebrow">THINK IT THROUGH</span>
        <span>✳</span>
      </div>
      <p className="companion-intro">
        Ask about this passage. Follow an idea across your reading.
      </p>
      <div className="companion-scope" aria-label="Answer sources">
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
      <p className="companion-index-status" role="status">
        {indexStatus}
      </p>
      <div className="companion-turns" aria-live="polite" aria-busy={busy}>
        {turns.map((turn, i) => (
          <article className="companion-turn" key={i}>
            <p className="companion-question">{turn.question}</p>
            <p className="companion-answer">
              {turn.answer.split(/(\[\d+\])/g).map((part, n) => {
                const match = part.match(/^\[(\d+)\]$/)
                const source =
                  match && turn.sources.find((s) => s.id === Number(match[1]))
                return source ? (
                  <button
                    type="button"
                    key={n}
                    className="companion-citation"
                    title={`${source.kind}: ${source.title}`}
                    onClick={() => onNavigate(source)}
                  >
                    {part}
                  </button>
                ) : (
                  part
                )
              })}
              {busy && i === turns.length - 1 && !turn.answer && (
                <span className="companion-thinking">
                  Reading the evidence…
                </span>
              )}
            </p>
            {!!turn.sources.length && (
              <details className="companion-evidence">
                <summary>Evidence · {turn.sources.length} passages</summary>
                {turn.sources.map((source) => (
                  <button
                    key={source.id}
                    type="button"
                    onClick={() => onNavigate(source)}
                  >
                    <strong>
                      [{source.id}]{' '}
                      {source.kind === 'thought' ? 'Your thought · ' : ''}
                      {source.title}
                    </strong>
                    <span>{source.text.slice(0, 180)}…</span>
                  </button>
                ))}
              </details>
            )}
            {turn.metrics && (
              <details className="companion-metrics">
                <summary>Response timings</summary>
                <pre>{JSON.stringify(turn.metrics, null, 2)}</pre>
              </details>
            )}
          </article>
        ))}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          void ask(question)
        }}
      >
        <label className="companion-input-label" htmlFor="companion-question">
          {voiceDraft
            ? 'Hearing you…'
            : 'Hold Space to speak · Esc leaves typing'}
        </label>
        <textarea
          id="companion-question"
          maxLength={2000}
          value={voiceDraft || question}
          readOnly={Boolean(voiceDraft)}
          className={voiceDraft ? 'is-listening' : undefined}
          aria-label={
            voiceDraft ? 'Live question transcription' : 'Your question'
          }
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What does the author mean here?"
          rows={2}
          onKeyDown={(event) => {
            if (event.key === 'Escape') event.currentTarget.blur()
          }}
        />
        <div className="companion-actions">
          <button type="submit" disabled={busy || !question.trim()}>
            Ask Margin ↗
          </button>
          {busy && (
            <button
              type="button"
              onClick={() => {
                controller.current?.abort()
                setBusy(false)
              }}
            >
              Stop
            </button>
          )}
        </div>
      </form>
      {error && (
        <p role="alert" className="voice-status has-error">
          {error}
        </p>
      )}
    </section>
  )
}
