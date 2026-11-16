import { useEffect, useState } from 'react'

type Section = { index: number; title: string }

export function ArticleOutline({ sections }: { sections: Section[] }) {
  const [active, setActive] = useState(0)
  const [hovered, setHovered] = useState<number | null>(null)
  useEffect(() => {
    let frame = 0
    const update = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        let current = 0
        for (let i = 0; i < sections.length; i++) {
          const element = document.querySelector(
            `[data-paragraph-index="${sections[i].index}"]`,
          )
          if (
            element &&
            element.getBoundingClientRect().top <= window.innerHeight * 0.3
          )
            current = i
        }
        setActive(current)
      })
    }
    update()
    window.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('scroll', update)
      window.removeEventListener('resize', update)
    }
  }, [sections])
  if (sections.length < 2) return null
  return (
    <nav
      className="article-outline"
      aria-label="Article sections"
      onMouseLeave={() => setHovered(null)}
    >
      <span className="outline-caption">In this article</span>
      <div className="outline-marks">
        {sections.map((section, i) => (
          <button
            type="button"
            key={section.index}
            className={`outline-mark${active === i ? ' is-current' : ''}`}
            aria-label={section.title}
            aria-current={active === i ? 'location' : undefined}
            onMouseEnter={() => setHovered(i)}
            onFocus={() => setHovered(i)}
            onBlur={() => setHovered(null)}
            onClick={() => {
              const element = document.querySelector<HTMLElement>(
                `[data-paragraph-index="${section.index}"]`,
              )
              element?.scrollIntoView({
                behavior: matchMedia('(prefers-reduced-motion: reduce)').matches
                  ? 'instant'
                  : 'smooth',
                block: 'start',
              })
              element?.focus({ preventScroll: true })
            }}
          >
            <span
              className="outline-dash"
              style={{
                transform: `scaleX(${hovered === null ? 1 : Math.max(1, 2.2 - Math.abs(hovered - i) * 0.4)})`,
              }}
            />
            <span className="outline-topic">{section.title}</span>
          </button>
        ))}
      </div>
    </nav>
  )
}
