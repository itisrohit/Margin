import { useState } from 'react'
import './App.css'

const NOTES_STORAGE_KEY = 'margin:sample-essay:notes'

const paragraphs = [
  'We tend to treat a thought as a thing that arrives finished. It appears, clear and self-contained, and we decide whether to keep it or let it go. But most ideas begin less like sentences and more like weather: a change in pressure, a small disturbance at the edge of attention.',
  'Reading gives those unfinished thoughts somewhere to land. A line in an essay can meet an experience we have not yet put into words. For a moment, the page and our own memory seem to be talking about the same thing.',
  'Usually we move on. The next paragraph arrives, then another, and the thought slips back into the noise of the day. We may remember that something mattered without remembering what it made us think.',
  'There is value in pausing before that happens. Not to turn every reading session into work, or to collect observations for their own sake. Just to leave a small trace: this part stayed with me; this is where my mind went.',
  'Over time, those traces begin to speak to one another. A thought from an article about attention might resurface beside a passage about friendship. The connection was not obvious when either piece was read. It becomes visible only because both were given a little room.',
  'Maybe that is all a margin needs to be: a place for the conversation between what we read and what we bring to it. The page holds the words. Beside them, there is room for whatever they set in motion.',
]

function loadNotes(): Record<number, string> {
  try {
    const stored = window.localStorage.getItem(NOTES_STORAGE_KEY)
    const parsed: unknown = stored ? JSON.parse(stored) : {}

    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(parsed).filter(
        ([index, note]) =>
          Number.isInteger(Number(index)) &&
          Number(index) >= 0 &&
          Number(index) < paragraphs.length &&
          typeof note === 'string',
      ),
    )
  } catch {
    return {}
  }
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      m.
    </span>
  )
}

function App() {
  const [activeParagraphIndex, setActiveParagraphIndex] = useState<
    number | null
  >(null)
  const [notes, setNotes] = useState<Record<number, string>>(loadNotes)
  const [noteDraft, setNoteDraft] = useState('')
  const [isEditingNote, setIsEditingNote] = useState(false)
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [storageWarning, setStorageWarning] = useState(false)

  const focusedParagraphIndex = activeParagraphIndex
  const focusedParagraph =
    focusedParagraphIndex === null ? null : paragraphs[focusedParagraphIndex]
  const activeNote =
    activeParagraphIndex === null ? undefined : notes[activeParagraphIndex]

  const persistNotes = (updatedNotes: Record<number, string>) => {
    setNotes(updatedNotes)
    try {
      window.localStorage.setItem(
        NOTES_STORAGE_KEY,
        JSON.stringify(updatedNotes),
      )
      setStorageWarning(false)
    } catch {
      setStorageWarning(true)
    }
  }

  const selectParagraph = (index: number) => {
    setActiveParagraphIndex(index)
    setNoteDraft(notes[index] ?? '')
    setIsEditingNote(false)
    setConfirmingDelete(false)
  }

  return (
    <div className={`app-shell${isEditingNote ? ' is-note-editing' : ''}`}>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Margin home">
          <BrandMark />
          <span>margin</span>
        </a>
        <span className="sample-badge">
          <span className="sample-dot" />
          Sample reading
        </span>
      </header>

      <main className="reader-layout" id="top">
        <article className="article">
          <div className="article-heading">
            <p className="eyebrow">A NOTE ON READING</p>
            <h1>The quiet value of an unfinished thought</h1>
            <p className="dek">
              Reading is more than taking in someone else’s words. It is a
              chance to notice what those words stir in you.
            </p>
            <div className="byline">
              <span className="byline-avatar" aria-hidden="true">
                M
              </span>
              <span className="byline-name">Margin sample</span>
              <span className="byline-separator" aria-hidden="true">
                ·
              </span>
              <span>4 min read</span>
              <span className="byline-separator" aria-hidden="true">
                ·
              </span>
              <span>Demo essay</span>
            </div>
          </div>

          <div className="article-body">
            {paragraphs.map((paragraph, index) => (
              <p
                key={paragraph}
                className={`article-paragraph${focusedParagraphIndex === index ? ' is-active' : ''}`}
                data-paragraph-index={index}
                tabIndex={0}
                onMouseEnter={() => {
                  if (!isEditingNote) selectParagraph(index)
                }}
                onClick={() => selectParagraph(index)}
                onFocus={() => selectParagraph(index)}
                onBlur={(event) => {
                  const reader = event.currentTarget.closest('.reader-layout')
                  if (!reader?.contains(event.relatedTarget as Node | null)) {
                    setActiveParagraphIndex(null)
                  }
                }}
              >
                {paragraph}
              </p>
            ))}
          </div>

          <footer className="article-footer">
            <span className="footer-rule" />
            <span>You’ve reached the end of this sample.</span>
            <span className="footer-rule" />
          </footer>
        </article>

        <aside className="margin-rail" aria-label="Margin notes">
          <div className="rail-heading">
            <span className="rail-label">
              {focusedParagraph ? 'IN FOCUS' : 'YOUR MARGIN'}
            </span>
            <span className="rail-count">
              {focusedParagraphIndex === null
                ? '—'
                : `${String(focusedParagraphIndex + 1).padStart(2, '0')} / ${String(paragraphs.length).padStart(2, '0')}`}
            </span>
          </div>
          <div className="rail-empty">
            <span className="rail-spark" aria-hidden="true">
              ✳
            </span>
            <p className="rail-title">
              {focusedParagraph
                ? `Passage ${focusedParagraphIndex! + 1}`
                : 'A little room to think.'}
            </p>
            <p className="rail-copy">
              {focusedParagraph
                ? `“${focusedParagraph.slice(0, 100)}${focusedParagraph.length > 100 ? '…' : ''}”`
                : 'Your thoughts will sit beside the words that sparked them.'}
            </p>
          </div>
          {activeParagraphIndex === null ? (
            <p className="rail-footnote">Select a passage to attach a note.</p>
          ) : activeNote && !isEditingNote ? (
            <div className="saved-note">
              <div className="saved-note-header">
                <p className="saved-note-label">YOUR THOUGHT</p>
                <span className="saved-note-status">SAVED</span>
              </div>
              <p className="saved-note-text">{activeNote}</p>
              <div className="note-actions">
                {confirmingDelete ? (
                  <>
                    <span className="delete-prompt">Delete this thought?</span>
                    <button
                      className="text-button"
                      type="button"
                      onClick={() => setConfirmingDelete(false)}
                    >
                      Keep
                    </button>
                    <button
                      className="text-button danger"
                      type="button"
                      onClick={() => {
                        const updated = { ...notes }
                        delete updated[activeParagraphIndex]
                        persistNotes(updated)
                        setNoteDraft('')
                        setConfirmingDelete(false)
                      }}
                    >
                      Delete
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      className="text-button"
                      type="button"
                      onClick={() => {
                        setNoteDraft(activeNote)
                        setIsEditingNote(true)
                      }}
                    >
                      Edit
                    </button>
                    <button
                      className="text-button danger"
                      type="button"
                      onClick={() => setConfirmingDelete(true)}
                    >
                      Delete
                    </button>
                  </>
                )}
              </div>
            </div>
          ) : isEditingNote ? (
            <form
              className="note-composer"
              onSubmit={(event) => {
                event.preventDefault()
                const text = noteDraft.trim()
                if (text && activeParagraphIndex !== null) {
                  persistNotes({
                    ...notes,
                    [activeParagraphIndex]: text,
                  })
                  setNoteDraft(text)
                  setIsEditingNote(false)
                }
              }}
            >
              <div className="note-composer-header">
                <label className="composer-label" htmlFor="note-draft">
                  <span className="composer-pencil" aria-hidden="true">
                    ✎
                  </span>
                  YOUR THOUGHT
                </label>
                <button
                  className="text-button"
                  type="button"
                  onClick={() => {
                    setNoteDraft(activeNote ?? '')
                    setIsEditingNote(false)
                  }}
                >
                  Cancel
                </button>
              </div>
              <textarea
                id="note-draft"
                autoFocus
                value={noteDraft}
                onChange={(event) => setNoteDraft(event.target.value)}
                placeholder="What did this passage make you think?"
                rows={5}
              />
              <div className="composer-footer">
                <p className="storage-note">
                  {storageWarning
                    ? 'Browser storage is unavailable.'
                    : 'This stays in your browser.'}
                </p>
                <button
                  className="save-note"
                  type="submit"
                  disabled={!noteDraft.trim()}
                >
                  Save thought <span aria-hidden="true">↗</span>
                </button>
              </div>
            </form>
          ) : (
            <button
              className="add-note"
              type="button"
              onClick={() => {
                setNoteDraft('')
                setIsEditingNote(true)
              }}
            >
              <span aria-hidden="true">✎</span> Leave a thought
            </button>
          )}
          <p className="rail-footnote">
            Notes stay with their passage on this device.
          </p>
        </aside>
      </main>

      <div className="keyboard-hint" aria-hidden="true">
        <span className="keyboard-key">SPACE</span>
        <span>to capture a thought</span>
        <span className="hint-soon">coming soon</span>
      </div>
    </div>
  )
}

export default App
