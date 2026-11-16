import './App.css'

const paragraphs = [
  'We tend to treat a thought as a thing that arrives finished. It appears, clear and self-contained, and we decide whether to keep it or let it go. But most ideas begin less like sentences and more like weather: a change in pressure, a small disturbance at the edge of attention.',
  'Reading gives those unfinished thoughts somewhere to land. A line in an essay can meet an experience we have not yet put into words. For a moment, the page and our own memory seem to be talking about the same thing.',
  'Usually we move on. The next paragraph arrives, then another, and the thought slips back into the noise of the day. We may remember that something mattered without remembering what it made us think.',
  'There is value in pausing before that happens. Not to turn every reading session into work, or to collect observations for their own sake. Just to leave a small trace: this part stayed with me; this is where my mind went.',
  'Over time, those traces begin to speak to one another. A thought from an article about attention might resurface beside a passage about friendship. The connection was not obvious when either piece was read. It becomes visible only because both were given a little room.',
  'Maybe that is all a margin needs to be: a place for the conversation between what we read and what we bring to it. The page holds the words. Beside them, there is room for whatever they set in motion.',
]

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      m.
    </span>
  )
}

function App() {
  return (
    <div className="app-shell">
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
            {paragraphs.map((paragraph) => (
              <p key={paragraph}>{paragraph}</p>
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
            <span className="rail-label">YOUR MARGIN</span>
            <span className="rail-count">—</span>
          </div>
          <div className="rail-empty">
            <span className="rail-spark" aria-hidden="true">
              ✳
            </span>
            <p className="rail-title">A little room to think.</p>
            <p className="rail-copy">
              Your thoughts will sit beside the words that sparked them.
            </p>
          </div>
          <p className="rail-footnote">
            This sample is here to explore the reading experience.
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
