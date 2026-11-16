import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

const NOTES_STORAGE_KEY = 'margin:sample-essay:notes'
const SPEECH_SETTINGS_KEY = 'margin:speech-settings'
const SPEECH_REVIEW_KEY = 'margin:sample-essay:speech-review'

const speechModels = [
  {
    id: 'cache-aware-parakeet-rnnt-en-US-asr-streaming-sortformer',
    name: 'Nemotron ASR',
    detail: 'English · Streaming model',
    mode: 'Words appear as you speak.',
  },
  {
    id: 'parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming-silero-vad-sortformer',
    name: 'Parakeet RNNT 1.1B',
    detail: '25 languages · Streaming model',
    mode: 'Words appear as you speak.',
  },
  {
    id: 'parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-offline-silero-vad-sortformer',
    name: 'Parakeet RNNT 1.1B',
    detail: '25 languages · Batch',
    mode: 'Returns transcription after recording finishes.',
  },
  {
    id: 'parakeet-tdt-0.6b-en-US-asr-offline',
    name: 'Parakeet TDT 0.6B',
    detail: 'English · Batch',
    mode: 'Returns transcription after recording finishes.',
  },
  {
    id: 'canary-1b-flash-multi-asr-offline',
    name: 'Canary 1B',
    detail: 'Multilingual · Batch',
    mode: 'Returns transcription after recording finishes.',
  },
] as const

type SpeechModelId = (typeof speechModels)[number]['id']
type SpeechSettings = { apiKey: string; modelId: SpeechModelId }
type TranscriptWord = {
  text: string
  start: number
  end: number
  confidence: number
}
type ServerTranscriptSegment = {
  text: string
  alternatives: string[]
  words: TranscriptWord[]
}
type TranscriptSegment = {
  text: string
  selected: string
  alternatives: string[]
  words: TranscriptWord[]
}
type TranscriptReview = {
  baseText: string
  segments: TranscriptSegment[]
}
type VoiceEditSelection = {
  start: number
  end: number
  anchorWord: number
  focusWord: number
}
type VoiceEditUndo = {
  paragraphIndex: number
  note: string
  review?: TranscriptReview
}
type LiveCapture = {
  socket: WebSocket
  stream: MediaStream
  context: AudioContext
  source: MediaStreamAudioSourceNode
  processor: ScriptProcessorNode
  mute: GainNode
  paragraphIndex: number
  latestText: string
}
const DEFAULT_SPEECH_MODEL_ID: SpeechModelId = speechModels[0].id
const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
const AUDIO_SAMPLE_RATE = 16_000

async function convertRecordingToWav(recording: Blob): Promise<ArrayBuffer> {
  const decoder = new AudioContext()
  try {
    const decoded = await decoder.decodeAudioData(await recording.arrayBuffer())
    const frameCount = Math.ceil(decoded.duration * AUDIO_SAMPLE_RATE)
    const renderer = new OfflineAudioContext(1, frameCount, AUDIO_SAMPLE_RATE)
    const source = renderer.createBufferSource()
    source.buffer = decoded
    source.connect(renderer.destination)
    source.start()
    const resampled = await renderer.startRendering()
    const samples = resampled.getChannelData(0)
    const wav = new ArrayBuffer(44 + samples.length * 2)
    const view = new DataView(wav)

    const writeString = (offset: number, value: string) => {
      for (let index = 0; index < value.length; index += 1) {
        view.setUint8(offset + index, value.charCodeAt(index))
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, wav.byteLength - 8, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, 1, true)
    view.setUint32(24, AUDIO_SAMPLE_RATE, true)
    view.setUint32(28, AUDIO_SAMPLE_RATE * 2, true)
    view.setUint16(32, 2, true)
    view.setUint16(34, 16, true)
    writeString(36, 'data')
    view.setUint32(40, samples.length * 2, true)

    for (let index = 0; index < samples.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, samples[index]))
      view.setInt16(
        44 + index * 2,
        sample < 0 ? sample * 0x8000 : sample * 0x7fff,
        true,
      )
    }

    return wav
  } finally {
    await decoder.close()
  }
}

function capitalizeSentenceStart(text: string): string {
  return text.replace(
    /^(\s*["'“‘([{]*)(\p{L})/u,
    (_, prefix, letter: string) => `${prefix}${letter.toUpperCase()}`,
  )
}

function appendTranscript(existingText: string, transcript: string): string {
  const existing = existingText.trimEnd()
  const incoming = transcript.trim().replace(/\s+/g, ' ')
  if (!existing) return capitalizeSentenceStart(incoming)
  if (!incoming) return existing

  const hasSentenceEnd = /[.!?…]["'’”)}\]]*$/u.test(existing)
  const continuesSameSentence = /[,;:—–-]["'’”)}\]]*$/u.test(existing)
  if (hasSentenceEnd) return `${existing} ${capitalizeSentenceStart(incoming)}`
  if (continuesSameSentence) return `${existing} ${incoming}`
  return `${existing}. ${capitalizeSentenceStart(incoming)}`
}

function loadSpeechSettings(): SpeechSettings {
  try {
    const stored = window.localStorage.getItem(SPEECH_SETTINGS_KEY)
    const parsed: unknown = stored ? JSON.parse(stored) : null
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      'apiKey' in parsed &&
      typeof parsed.apiKey === 'string'
    ) {
      const modelId =
        'modelId' in parsed &&
        speechModels.some((model) => model.id === parsed.modelId)
          ? (parsed.modelId as SpeechModelId)
          : DEFAULT_SPEECH_MODEL_ID
      return { apiKey: parsed.apiKey, modelId }
    }
  } catch {
    // Settings are optional; use the defaults if browser storage is unavailable.
  }
  return { apiKey: '', modelId: DEFAULT_SPEECH_MODEL_ID }
}

function loadTranscriptReviews(): Record<number, TranscriptReview> {
  try {
    const stored = window.localStorage.getItem(SPEECH_REVIEW_KEY)
    const parsed: unknown = stored ? JSON.parse(stored) : {}
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      return {}
    }
    return Object.fromEntries(
      Object.entries(parsed).flatMap(([index, value]) => {
        if (
          !Number.isInteger(Number(index)) ||
          Number(index) < 0 ||
          Number(index) >= paragraphs.length ||
          typeof value !== 'object' ||
          value === null ||
          !('baseText' in value) ||
          typeof value.baseText !== 'string' ||
          !('segments' in value) ||
          !Array.isArray(value.segments)
        ) {
          return []
        }
        const segments = value.segments.filter(
          (segment: unknown): segment is TranscriptSegment =>
            typeof segment === 'object' &&
            segment !== null &&
            'text' in segment &&
            typeof segment.text === 'string' &&
            'selected' in segment &&
            typeof segment.selected === 'string' &&
            'alternatives' in segment &&
            Array.isArray(segment.alternatives) &&
            segment.alternatives.every(
              (alternative: unknown) => typeof alternative === 'string',
            ) &&
            'words' in segment &&
            Array.isArray(segment.words) &&
            segment.words.every(
              (word: unknown) =>
                typeof word === 'object' &&
                word !== null &&
                'text' in word &&
                typeof word.text === 'string' &&
                'start' in word &&
                typeof word.start === 'number' &&
                'end' in word &&
                typeof word.end === 'number' &&
                'confidence' in word &&
                typeof word.confidence === 'number',
            ),
        )
        return [[Number(index), { baseText: value.baseText, segments }]]
      }),
    )
  } catch {
    return {}
  }
}

function normalizeToken(token: string): string {
  return token
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '')
    .toLocaleLowerCase()
}

function alignedAlternativeToken(
  currentTokens: string[],
  alternativeTokens: string[],
  targetIndex: number,
): string | undefined {
  const rows = currentTokens.length + 1
  const columns = alternativeTokens.length + 1
  const costs = Array.from({ length: rows }, () =>
    Array<number>(columns).fill(0),
  )
  for (let row = 0; row < rows; row += 1) costs[row][0] = row
  for (let column = 0; column < columns; column += 1) costs[0][column] = column
  for (let row = 1; row < rows; row += 1) {
    for (let column = 1; column < columns; column += 1) {
      const substitution =
        normalizeToken(currentTokens[row - 1]) ===
        normalizeToken(alternativeTokens[column - 1])
          ? 0
          : 1
      costs[row][column] = Math.min(
        costs[row - 1][column] + 1,
        costs[row][column - 1] + 1,
        costs[row - 1][column - 1] + substitution,
      )
    }
  }

  let row = currentTokens.length
  let column = alternativeTokens.length
  while (row > 0 || column > 0) {
    if (
      row > 0 &&
      column > 0 &&
      costs[row][column] ===
        costs[row - 1][column - 1] +
          (normalizeToken(currentTokens[row - 1]) ===
          normalizeToken(alternativeTokens[column - 1])
            ? 0
            : 1)
    ) {
      if (row - 1 === targetIndex) {
        return alternativeTokens[column - 1]
      }
      row -= 1
      column -= 1
    } else if (row > 0 && costs[row][column] === costs[row - 1][column] + 1) {
      if (row - 1 === targetIndex) return undefined
      row -= 1
    } else {
      column -= 1
    }
  }
  return undefined
}

function tokenAlternatives(
  segment: TranscriptSegment,
  tokenIndex: number,
): string[] {
  const currentTokens = segment.selected.split(/\s+/u)
  const current = currentTokens[tokenIndex]
  if (!current) return []
  return Array.from(
    new Set(
      segment.alternatives
        .map((alternative) =>
          alignedAlternativeToken(
            currentTokens,
            alternative.split(/\s+/u),
            tokenIndex,
          ),
        )
        .filter((token): token is string => Boolean(token))
        .filter((token) => normalizeToken(token) !== normalizeToken(current)),
    ),
  )
}

function prepareReviewSegments(
  segments: ServerTranscriptSegment[],
): TranscriptSegment[] {
  return segments.map((segment) => ({
    text: segment.text,
    selected: segment.text,
    alternatives: Array.from(
      new Set(
        segment.alternatives
          .map((alternative) => alternative.trim())
          .filter(Boolean),
      ),
    ),
    words: segment.words ?? [],
  }))
}

function noteWordRanges(
  text: string,
): Array<{ text: string; start: number; end: number }> {
  return Array.from(text.matchAll(/\S+/gu), (match) => ({
    text: match[0],
    start: match.index,
    end: match.index + match[0].length,
  }))
}

function replaceTextRange(
  text: string,
  start: number,
  end: number,
  replacement: string,
): string {
  const before = text.slice(0, start)
  const selected = text.slice(start, end)
  const after = text.slice(end)
  let spoken = replacement.trim().replace(/\s+/gu, ' ')
  const sentenceEnd = /[.!?…]["'’”)}\]]*$/u
  const originalEnding = selected.match(sentenceEnd)?.[0] ?? ''

  // Recognition punctuation belongs to the sentence context, not the spoken
  // fragment. Keep the original boundary punctuation and avoid inserting a
  // sentence break in the middle of the note.
  if (after.trim() && !originalEnding) {
    spoken = spoken.replace(sentenceEnd, '')
  } else if (originalEnding) {
    spoken = spoken.replace(sentenceEnd, '') + originalEnding
  }

  const nextText =
    after && originalEnding ? capitalizeSentenceStart(after) : after
  return `${before}${spoken}${nextText}`
}

function locateReviewRange(
  review: TranscriptReview,
  start: number,
  end: number,
): { segmentIndex: number; localStart: number; localEnd: number } | null {
  const fullTranscript = appendTranscript(
    review.baseText,
    review.segments.map((segment) => segment.selected).join(' '),
  )
  let searchFrom = review.baseText.length
  for (const [segmentIndex, segment] of review.segments.entries()) {
    const segmentStart = fullTranscript.indexOf(segment.selected, searchFrom)
    if (segmentStart < 0) continue
    const localStart = start - segmentStart
    const localEnd = end - segmentStart
    if (localStart >= 0 && localEnd <= segment.selected.length) {
      return { segmentIndex, localStart, localEnd }
    }
    searchFrom = segmentStart + segment.selected.length
  }
  return null
}

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
  const [transcriptReviews, setTranscriptReviews] = useState<
    Record<number, TranscriptReview>
  >(loadTranscriptReviews)
  const [openCorrection, setOpenCorrection] = useState<{
    segmentIndex: number
    wordIndex: number
  } | null>(null)
  const [isVoiceEditing, setIsVoiceEditing] = useState(false)
  const [voiceEditSelection, setVoiceEditSelection] =
    useState<VoiceEditSelection | null>(null)
  const [isExtendingSelection, setIsExtendingSelection] = useState(false)
  const [voiceEditUndo, setVoiceEditUndo] = useState<VoiceEditUndo | null>(null)
  const [noteDraft, setNoteDraft] = useState('')
  const [isEditingNote, setIsEditingNote] = useState(false)
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [storageWarning, setStorageWarning] = useState(false)
  const [speechSettings, setSpeechSettings] =
    useState<SpeechSettings>(loadSpeechSettings)
  const [settingsDraft, setSettingsDraft] = useState<SpeechSettings>({
    apiKey: '',
    modelId: DEFAULT_SPEECH_MODEL_ID,
  })
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [settingsSaved, setSettingsSaved] = useState(false)
  const [settingsStorageError, setSettingsStorageError] = useState(false)
  const [voiceState, setVoiceState] = useState<
    'idle' | 'recording' | 'transcribing' | 'error'
  >('idle')
  const [isLivePreview, setIsLivePreview] = useState(false)
  const [voiceError, setVoiceError] = useState('')
  const recorderRef = useRef<MediaRecorder | null>(null)
  const liveCaptureRef = useRef<LiveCapture | null>(null)
  const voiceHoldRef = useRef(false)
  const holdSourceRef = useRef<'keyboard' | 'pointer' | null>(null)

  const focusedParagraphIndex = activeParagraphIndex
  const focusedParagraph =
    focusedParagraphIndex === null ? null : paragraphs[focusedParagraphIndex]
  const activeNote =
    activeParagraphIndex === null ? undefined : notes[activeParagraphIndex]
  const activeReview =
    activeParagraphIndex === null
      ? undefined
      : transcriptReviews[activeParagraphIndex]

  const persistNotes = useCallback((updatedNotes: Record<number, string>) => {
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
  }, [])

  const persistTranscriptReviews = useCallback(
    (updatedReviews: Record<number, TranscriptReview>) => {
      setTranscriptReviews(updatedReviews)
      try {
        window.localStorage.setItem(
          SPEECH_REVIEW_KEY,
          JSON.stringify(updatedReviews),
        )
      } catch {
        setStorageWarning(true)
      }
    },
    [],
  )

  const chooseAlternative = (segmentIndex: number, alternative: string) => {
    if (activeParagraphIndex === null) return
    const review = transcriptReviews[activeParagraphIndex]
    if (!review) return
    const segments = review.segments.map((segment, index) =>
      index === segmentIndex ? { ...segment, selected: alternative } : segment,
    )
    const updatedReview = { ...review, segments }
    const transcript = segments.map((segment) => segment.selected).join(' ')
    const updatedNote = appendTranscript(review.baseText, transcript)
    persistNotes({ ...notes, [activeParagraphIndex]: updatedNote })
    setNoteDraft(updatedNote)
    persistTranscriptReviews({
      ...transcriptReviews,
      [activeParagraphIndex]: updatedReview,
    })
    setOpenCorrection(null)
  }

  const renderTranscript = (review: TranscriptReview) => (
    <div className="saved-note-text">
      {review.baseText && <span>{review.baseText} </span>}
      {review.segments.map((segment, segmentIndex) => {
        const words =
          segment.selected === segment.text && segment.words.length
            ? segment.words.map((word) => word.text)
            : segment.selected.split(/\s+/u)
        return (
          <span key={`${segmentIndex}-${segment.text}`}>
            {segmentIndex > 0 && ' '}
            {words.map((word, wordIndex) => {
              const alternatives = tokenAlternatives(segment, wordIndex)
              return (
                <span key={`${wordIndex}-${word}`}>
                  {wordIndex > 0 && ' '}
                  {alternatives.length ? (
                    <button
                      className="transcript-word has-alternatives"
                      type="button"
                      aria-label={`Review the transcription of ${word}`}
                      onClick={() =>
                        setOpenCorrection({ segmentIndex, wordIndex })
                      }
                    >
                      {word}
                    </button>
                  ) : (
                    word
                  )}
                </span>
              )
            })}
          </span>
        )
      })}
    </div>
  )

  const activeCorrectionSegment =
    openCorrection && activeReview
      ? activeReview.segments[openCorrection.segmentIndex]
      : undefined
  const voiceEditReviewLocation =
    voiceEditSelection && activeReview
      ? locateReviewRange(
          activeReview,
          voiceEditSelection.start,
          voiceEditSelection.end,
        )
      : null
  const voiceEditReviewSegment = voiceEditReviewLocation
    ? activeReview?.segments[voiceEditReviewLocation.segmentIndex]
    : undefined
  const voiceEditSegmentWordIndex = voiceEditReviewSegment
    ? noteWordRanges(voiceEditReviewSegment.selected).findIndex(
        (word) =>
          word.start < voiceEditReviewLocation!.localEnd &&
          word.end > voiceEditReviewLocation!.localStart,
      )
    : -1
  const voiceEditAlternatives =
    voiceEditReviewSegment &&
    voiceEditSegmentWordIndex >= 0 &&
    voiceEditSelection &&
    activeNote
      ?.slice(voiceEditSelection.start, voiceEditSelection.end)
      .trim()
      .split(/\s+/u).length === 1
      ? tokenAlternatives(voiceEditReviewSegment, voiceEditSegmentWordIndex)
      : []

  const selectParagraph = (index: number) => {
    setActiveParagraphIndex(index)
    setNoteDraft(notes[index] ?? '')
    setIsEditingNote(false)
    setConfirmingDelete(false)
    setOpenCorrection(null)
    setIsVoiceEditing(false)
    setVoiceEditSelection(null)
    setVoiceEditUndo(null)
  }

  const selectVoiceWord = (wordIndex: number, extend = false) => {
    const ranges = noteWordRanges(activeNote ?? '')
    const range = ranges[wordIndex]
    if (!range) return
    const current = voiceEditSelection
    const shouldExtend = extend || isExtendingSelection
    const anchorWord = shouldExtend && current ? current.anchorWord : wordIndex
    const firstWord = ranges[Math.min(anchorWord, wordIndex)]
    const lastWord = ranges[Math.max(anchorWord, wordIndex)]
    setVoiceEditSelection({
      start: firstWord.start,
      end: lastWord.end,
      anchorWord,
      focusWord: wordIndex,
    })
    setOpenCorrection(null)
    if (shouldExtend && current) setIsExtendingSelection(false)
  }

  const renderVoiceEditWords = (text: string) => (
    <div className="saved-note-text voice-edit-text">
      {noteWordRanges(text).map((word, index) => (
        <span key={`${word.start}-${word.text}`}>
          {index > 0 && ' '}
          <button
            className={`voice-edit-word${
              voiceEditSelection &&
              word.start >= voiceEditSelection.start &&
              word.end <= voiceEditSelection.end
                ? ' is-selected'
                : ''
            }`}
            id={`voice-edit-word-${index}`}
            type="button"
            aria-pressed={Boolean(
              voiceEditSelection &&
              word.start >= voiceEditSelection.start &&
              word.end <= voiceEditSelection.end,
            )}
            onClick={(event) => selectVoiceWord(index, event.shiftKey)}
            onKeyDown={(event) => {
              if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight')
                return
              const nextIndex = index + (event.key === 'ArrowRight' ? 1 : -1)
              if (nextIndex < 0 || nextIndex >= noteWordRanges(text).length)
                return
              event.preventDefault()
              selectVoiceWord(nextIndex, event.shiftKey)
              document.getElementById(`voice-edit-word-${nextIndex}`)?.focus()
            }}
          >
            {word.text}
          </button>
        </span>
      ))}
    </div>
  )

  const applyVoiceEditReplacement = useCallback(
    (replacement: string) => {
      if (activeParagraphIndex === null || !activeNote || !voiceEditSelection) {
        return
      }
      const paragraphIndex = activeParagraphIndex
      const previousReview = transcriptReviews[paragraphIndex]
      let updatedNote = replaceTextRange(
        activeNote,
        voiceEditSelection.start,
        voiceEditSelection.end,
        replacement,
      )
      let updatedReview: TranscriptReview | undefined

      if (previousReview) {
        const location = locateReviewRange(
          previousReview,
          voiceEditSelection.start,
          voiceEditSelection.end,
        )
        if (location) {
          const segment = previousReview.segments[location.segmentIndex]
          const oldWordRanges = noteWordRanges(segment.selected)
          const firstWordIndex = oldWordRanges.findIndex(
            (word) => word.end > location.localStart,
          )
          const lastWordIndex = oldWordRanges.findLastIndex(
            (word) => word.start < location.localEnd,
          )
          const replacementWords = replacement
            .trim()
            .split(/\s+/u)
            .filter(Boolean)
          let words = segment.words
          if (firstWordIndex >= 0 && lastWordIndex >= firstWordIndex) {
            const timingWords = segment.words.slice(
              firstWordIndex,
              lastWordIndex + 1,
            )
            const fallbackTiming = timingWords[0]
            const replacementMetadata = replacementWords.map((text, index) => {
              const timing =
                timingWords[Math.min(index, timingWords.length - 1)] ??
                fallbackTiming
              return {
                text,
                start: timing?.start ?? 0,
                end: timing?.end ?? 0,
                confidence: timing?.confidence ?? 0,
              }
            })
            words = [
              ...segment.words.slice(0, firstWordIndex),
              ...replacementMetadata,
              ...segment.words.slice(lastWordIndex + 1),
            ]
          }
          const selected = replaceTextRange(
            segment.selected,
            location.localStart,
            location.localEnd,
            replacement,
          )
          const segments = previousReview.segments.map((item, index) =>
            index === location.segmentIndex
              ? {
                  ...item,
                  selected,
                  alternatives: Array.from(
                    new Set([...item.alternatives, selected]),
                  ),
                  words,
                }
              : item,
          )
          updatedReview = { ...previousReview, segments }
          updatedNote = appendTranscript(
            updatedReview.baseText,
            segments.map((item) => item.selected).join(' '),
          )
        }
      }

      setVoiceEditUndo({
        paragraphIndex,
        note: activeNote,
        review: previousReview,
      })
      persistNotes({ ...notes, [paragraphIndex]: updatedNote })
      setNoteDraft(updatedNote)
      const reviews = { ...transcriptReviews }
      if (updatedReview) reviews[paragraphIndex] = updatedReview
      else delete reviews[paragraphIndex]
      persistTranscriptReviews(reviews)
      setVoiceEditSelection(null)
      setIsExtendingSelection(false)
      setOpenCorrection(null)
    },
    [
      activeNote,
      activeParagraphIndex,
      notes,
      persistNotes,
      persistTranscriptReviews,
      transcriptReviews,
      voiceEditSelection,
    ],
  )

  const undoVoiceEdit = () => {
    if (
      !voiceEditUndo ||
      activeParagraphIndex !== voiceEditUndo.paragraphIndex
    ) {
      return
    }
    persistNotes({ ...notes, [activeParagraphIndex]: voiceEditUndo.note })
    setNoteDraft(voiceEditUndo.note)
    const reviews = { ...transcriptReviews }
    if (voiceEditUndo.review)
      reviews[activeParagraphIndex] = voiceEditUndo.review
    else delete reviews[activeParagraphIndex]
    persistTranscriptReviews(reviews)
    setVoiceEditUndo(null)
    setVoiceEditSelection(null)
  }

  const startVoiceCapture = useCallback(async () => {
    if (recorderRef.current || !voiceHoldRef.current) return
    if (isVoiceEditing && !voiceEditSelection) {
      setVoiceError('Tap a word first to choose what to replace.')
      voiceHoldRef.current = false
      return
    }
    if (activeParagraphIndex === null) {
      setVoiceError('Select a passage before recording a thought.')
      setVoiceState('error')
      voiceHoldRef.current = false
      return
    }
    if (!speechSettings.apiKey) {
      setSettingsDraft(speechSettings)
      setIsSettingsOpen(true)
      voiceHoldRef.current = false
      return
    }

    const paragraphIndex = activeParagraphIndex
    let stream: MediaStream | null = null
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('This browser does not support microphone access.')
      }
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1 },
      })
      if (!voiceHoldRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }

      const model = speechModels.find(
        (candidate) => candidate.id === speechSettings.modelId,
      )
      if (model?.detail.includes('Streaming')) {
        const websocketUrl = new URL(API_URL)
        websocketUrl.protocol =
          websocketUrl.protocol === 'https:' ? 'wss:' : 'ws:'
        websocketUrl.pathname = '/speech/stream'
        const socket = new WebSocket(websocketUrl)
        socket.binaryType = 'arraybuffer'
        await new Promise<void>((resolve, reject) => {
          socket.onopen = () => resolve()
          socket.onerror = () =>
            reject(new Error('Could not connect to transcription.'))
        })
        if (!voiceHoldRef.current) {
          socket.close()
          stream.getTracks().forEach((track) => track.stop())
          return
        }

        const context = new AudioContext()
        await context.resume()
        if (!voiceHoldRef.current) {
          socket.close()
          void context.close()
          stream.getTracks().forEach((track) => track.stop())
          return
        }
        const source = context.createMediaStreamSource(stream)
        const processor = context.createScriptProcessor(4096, 1, 1)
        const mute = context.createGain()
        mute.gain.value = 0
        const capture: LiveCapture = {
          socket,
          stream,
          context,
          source,
          processor,
          mute,
          paragraphIndex,
          latestText: '',
        }
        liveCaptureRef.current = capture
        socket.onmessage = (event: MessageEvent<string>) => {
          const result = JSON.parse(event.data) as {
            type: 'transcript' | 'done' | 'error'
            text?: string
            detail?: string
            segments?: ServerTranscriptSegment[]
          }
          if (result.type === 'transcript' && result.text) {
            capture.latestText = result.text
            setNoteDraft(
              isVoiceEditing && voiceEditSelection
                ? replaceTextRange(
                    notes[paragraphIndex] ?? '',
                    voiceEditSelection.start,
                    voiceEditSelection.end,
                    result.text,
                  )
                : appendTranscript(notes[paragraphIndex] ?? '', result.text),
            )
          } else if (result.type === 'done') {
            const transcript = (result.text || capture.latestText).trim()
            if (transcript) {
              if (isVoiceEditing && voiceEditSelection) {
                applyVoiceEditReplacement(transcript)
              } else {
                const updatedNote = appendTranscript(
                  notes[paragraphIndex] ?? '',
                  transcript,
                )
                persistNotes({ ...notes, [paragraphIndex]: updatedNote })
                setNoteDraft(updatedNote)
                const segments = prepareReviewSegments(result.segments ?? [])
                if (segments.length) {
                  persistTranscriptReviews({
                    ...transcriptReviews,
                    [paragraphIndex]: {
                      baseText: notes[paragraphIndex] ?? '',
                      segments,
                    },
                  })
                }
              }
              setVoiceError('')
            } else {
              setVoiceError(
                'No speech was detected. Try speaking a little longer.',
              )
              setVoiceState('error')
              setIsLivePreview(false)
              liveCaptureRef.current = null
              socket.close()
              void context.close()
              capture.stream.getTracks().forEach((track) => track.stop())
              return
            }
            setVoiceState('idle')
            setIsLivePreview(false)
            liveCaptureRef.current = null
            socket.close()
            void context.close()
            capture.stream.getTracks().forEach((track) => track.stop())
          } else if (result.type === 'error') {
            setVoiceError(result.detail ?? 'Transcription failed.')
            setVoiceState('error')
            setIsLivePreview(false)
            liveCaptureRef.current = null
            socket.close()
            void context.close()
            capture.stream.getTracks().forEach((track) => track.stop())
          }
        }
        socket.onclose = () => {
          if (liveCaptureRef.current === capture) {
            setVoiceError('The live transcription connection closed.')
            setVoiceState('error')
            setIsLivePreview(false)
            liveCaptureRef.current = null
            void context.close()
            capture.stream.getTracks().forEach((track) => track.stop())
          }
        }
        socket.send(
          JSON.stringify({
            apiKey: speechSettings.apiKey,
            modelId: speechSettings.modelId,
            correctionMode: isVoiceEditing,
          }),
        )
        processor.onaudioprocess = (event) => {
          if (socket.readyState !== WebSocket.OPEN) return
          const input = event.inputBuffer.getChannelData(0)
          const ratio = context.sampleRate / AUDIO_SAMPLE_RATE
          const outputLength = Math.floor(input.length / ratio)
          const pcm = new Int16Array(outputLength)
          for (let index = 0; index < outputLength; index += 1) {
            const sample = input[Math.floor(index * ratio)]
            const bounded = Math.max(-1, Math.min(1, sample))
            pcm[index] = bounded < 0 ? bounded * 0x8000 : bounded * 0x7fff
          }
          socket.send(pcm.buffer)
        }
        source.connect(processor)
        processor.connect(mute)
        mute.connect(context.destination)
        setVoiceError('')
        setIsLivePreview(true)
        setVoiceState('recording')
        return
      }

      if (!window.MediaRecorder) {
        throw new Error('This browser does not support microphone recording.')
      }
      const preferredType = [
        'audio/webm;codecs=opus',
        'audio/ogg;codecs=opus',
        'audio/mp4',
      ].find((type) => MediaRecorder.isTypeSupported(type))
      const recorder = new MediaRecorder(
        stream,
        preferredType ? { mimeType: preferredType } : undefined,
      )
      const chunks: Blob[] = []
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data)
      }
      recorder.onstop = () => {
        stream?.getTracks().forEach((track) => track.stop())
        void (async () => {
          setVoiceState('transcribing')
          try {
            const wav = await convertRecordingToWav(
              new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }),
            )
            const response = await fetch(`${API_URL}/speech/transcribe`, {
              method: 'POST',
              headers: {
                Authorization: `Bearer ${speechSettings.apiKey}`,
                'Content-Type': 'audio/wav',
                'X-Speech-Model': speechSettings.modelId,
                'X-Speech-Correction': String(isVoiceEditing),
              },
              body: wav,
            })
            const result: {
              text?: string
              detail?: string
              segments?: ServerTranscriptSegment[]
            } = await response.json()
            if (!response.ok)
              throw new Error(result.detail ?? 'Transcription failed.')
            const transcript = result.text?.trim()
            if (!transcript)
              throw new Error(
                'No speech was detected. Try speaking a little longer.',
              )

            const existingNote = notes[paragraphIndex] ?? ''
            if (isVoiceEditing && voiceEditSelection) {
              applyVoiceEditReplacement(transcript)
            } else {
              const updatedNote = appendTranscript(existingNote, transcript)
              persistNotes({ ...notes, [paragraphIndex]: updatedNote })
              setNoteDraft(updatedNote)
              const segments = prepareReviewSegments(result.segments ?? [])
              if (segments.length) {
                persistTranscriptReviews({
                  ...transcriptReviews,
                  [paragraphIndex]: { baseText: existingNote, segments },
                })
              }
            }
            setVoiceError('')
            setVoiceState('idle')
          } catch (error) {
            setVoiceError(
              error instanceof Error ? error.message : 'Transcription failed.',
            )
            setVoiceState('error')
          } finally {
            recorderRef.current = null
          }
        })()
      }
      recorder.onerror = () => {
        stream?.getTracks().forEach((track) => track.stop())
        recorderRef.current = null
        setVoiceError('The microphone recording stopped unexpectedly.')
        setVoiceState('error')
      }

      recorderRef.current = recorder
      recorder.start()
      setVoiceError('')
      setVoiceState('recording')
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop())
      voiceHoldRef.current = false
      setVoiceError(
        error instanceof Error && error.message.includes('does not support')
          ? error.message
          : 'Allow microphone access to record a thought.',
      )
      setVoiceState('error')
    }
  }, [
    activeParagraphIndex,
    applyVoiceEditReplacement,
    isVoiceEditing,
    notes,
    persistNotes,
    persistTranscriptReviews,
    speechSettings,
    transcriptReviews,
    voiceEditSelection,
  ])

  const stopVoiceCapture = useCallback(() => {
    voiceHoldRef.current = false
    const liveCapture = liveCaptureRef.current
    if (liveCapture) {
      setVoiceState('transcribing')
      liveCapture.source.disconnect()
      liveCapture.processor.disconnect()
      liveCapture.mute.disconnect()
      liveCapture.stream.getTracks().forEach((track) => track.stop())
      if (liveCapture.socket.readyState === WebSocket.OPEN) {
        liveCapture.socket.send('stop')
      }
      void liveCapture.context.close()
      return
    }
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') recorder.stop()
  }, [])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space') return
      const target = event.target
      if (
        target instanceof HTMLElement &&
        target.closest(
          'input, textarea, select, [contenteditable="true"], button:not(.voice-capture):not(.voice-edit-word)',
        )
      )
        return
      event.preventDefault()
      if (
        event.repeat ||
        isEditingNote ||
        (isVoiceEditing && !voiceEditSelection) ||
        isSettingsOpen ||
        voiceState === 'recording' ||
        voiceState === 'transcribing'
      )
        return
      voiceHoldRef.current = true
      holdSourceRef.current = 'keyboard'
      void startVoiceCapture()
    }
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space' && holdSourceRef.current === 'keyboard') {
        holdSourceRef.current = null
        stopVoiceCapture()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [
    isEditingNote,
    isVoiceEditing,
    isSettingsOpen,
    startVoiceCapture,
    stopVoiceCapture,
    voiceState,
    voiceEditSelection,
  ])

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
        <button
          className="settings-trigger"
          type="button"
          aria-label="Open settings"
          onClick={() => {
            setSettingsDraft(speechSettings)
            setSettingsSaved(false)
            setSettingsStorageError(false)
            setIsSettingsOpen(true)
          }}
        >
          <span aria-hidden="true">⚙</span>
          Settings
          {speechSettings.apiKey && (
            <span className="settings-indicator" aria-label="API key saved" />
          )}
        </button>
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
          {activeParagraphIndex !== null && (
            <button
              className={`voice-capture${voiceState === 'recording' ? ' is-recording' : ''}`}
              type="button"
              aria-pressed={voiceState === 'recording'}
              disabled={
                voiceState === 'transcribing' ||
                isEditingNote ||
                (isVoiceEditing && !voiceEditSelection)
              }
              onPointerDown={(event) => {
                event.preventDefault()
                if (!speechSettings.apiKey) {
                  setSettingsDraft(speechSettings)
                  setIsSettingsOpen(true)
                  return
                }
                voiceHoldRef.current = true
                holdSourceRef.current = 'pointer'
                void startVoiceCapture()
              }}
              onPointerUp={() => {
                if (holdSourceRef.current === 'pointer') {
                  holdSourceRef.current = null
                  stopVoiceCapture()
                }
              }}
              onPointerLeave={() => {
                if (holdSourceRef.current === 'pointer') {
                  holdSourceRef.current = null
                  stopVoiceCapture()
                }
              }}
              onPointerCancel={() => {
                if (holdSourceRef.current === 'pointer') {
                  holdSourceRef.current = null
                  stopVoiceCapture()
                }
              }}
            >
              <span className="voice-indicator" aria-hidden="true" />
              {voiceState === 'recording'
                ? isVoiceEditing
                  ? 'Release to replace selection'
                  : 'Release to finish thought'
                : voiceState === 'transcribing'
                  ? 'Transcribing…'
                  : isVoiceEditing
                    ? voiceEditSelection
                      ? 'Hold to replace selection'
                      : 'Choose a word to edit'
                    : speechSettings.apiKey
                      ? 'Hold to speak a thought'
                      : 'Add API key to speak'}
            </button>
          )}
          {voiceState !== 'idle' && (
            <div
              className={`voice-status${voiceState === 'error' ? ' has-error' : ''}`}
              role="status"
            >
              {voiceState === 'recording'
                ? isVoiceEditing
                  ? 'Listening — release to replace the selected words.'
                  : 'Listening — release to finish your thought.'
                : voiceState === 'transcribing'
                  ? 'Writing your thought in the margin…'
                  : voiceError}
            </div>
          )}
          {activeParagraphIndex === null ? (
            <p className="rail-footnote">Select a passage to attach a note.</p>
          ) : isLivePreview && noteDraft ? (
            <div className="saved-note live-note">
              <div className="saved-note-header">
                <p className="saved-note-label">
                  {isVoiceEditing ? 'VOICE EDIT' : 'YOUR THOUGHT'}
                </p>
                <span className="saved-note-status">WRITING</span>
              </div>
              <p className="saved-note-text">{noteDraft}</p>
            </div>
          ) : activeNote && !isEditingNote ? (
            <div className="saved-note">
              <div className="saved-note-header">
                <p className="saved-note-label">
                  {isVoiceEditing ? 'EDITING BY VOICE' : 'YOUR THOUGHT'}
                </p>
                <span className="saved-note-status">
                  {isVoiceEditing ? 'SELECT A WORD' : 'SAVED'}
                </span>
              </div>
              {isVoiceEditing ? (
                <p className="review-note-hint">
                  Tap a word. Choose “Extend to phrase” to include more words,
                  or use Shift and the arrow keys.
                </p>
              ) : (
                activeReview?.segments.some((segment) =>
                  segment.alternatives.some(
                    (alternative) => alternative !== segment.selected,
                  ),
                ) && (
                  <p className="review-note-hint">
                    Underlined words have another take. Tap one to choose a
                    version.
                  </p>
                )
              )}
              {isVoiceEditing ? (
                renderVoiceEditWords(activeNote)
              ) : activeReview ? (
                renderTranscript(activeReview)
              ) : (
                <p className="saved-note-text">{activeNote}</p>
              )}
              {isVoiceEditing && (
                <div className="voice-edit-panel">
                  <div className="voice-edit-panel-header">
                    <p className="correction-label">
                      {voiceEditSelection
                        ? `SELECTED: ${activeNote.slice(voiceEditSelection.start, voiceEditSelection.end)}`
                        : 'CHOOSE A WORD OR PHRASE'}
                    </p>
                    <button
                      className={`text-button${isExtendingSelection ? ' is-active' : ''}`}
                      type="button"
                      aria-pressed={isExtendingSelection}
                      onClick={() =>
                        setIsExtendingSelection((current) => !current)
                      }
                    >
                      {isExtendingSelection
                        ? 'Tap end of phrase'
                        : 'Extend to phrase'}
                    </button>
                  </div>
                  {voiceEditSelection && voiceEditAlternatives.length > 0 && (
                    <div className="correction-options">
                      <p className="correction-hint">
                        Another way NVIDIA heard it
                      </p>
                      {voiceEditAlternatives.map((alternative) => (
                        <button
                          className="correction-option"
                          type="button"
                          key={alternative}
                          onClick={() => applyVoiceEditReplacement(alternative)}
                        >
                          {alternative}
                        </button>
                      ))}
                    </div>
                  )}
                  {voiceEditUndo?.paragraphIndex === activeParagraphIndex && (
                    <button
                      className="text-button voice-edit-undo"
                      type="button"
                      onClick={undoVoiceEdit}
                    >
                      Undo last change
                    </button>
                  )}
                  <div className="voice-edit-actions">
                    <button
                      className="text-button"
                      type="button"
                      onClick={() => {
                        setNoteDraft(activeNote)
                        setIsVoiceEditing(false)
                        setVoiceEditSelection(null)
                        setIsExtendingSelection(false)
                      }}
                    >
                      Done
                    </button>
                    <button
                      className="text-button text-button-subtle"
                      type="button"
                      onClick={() => {
                        setNoteDraft(activeNote)
                        setIsVoiceEditing(false)
                        setVoiceEditSelection(null)
                        setIsEditingNote(true)
                      }}
                    >
                      Edit with keyboard
                    </button>
                  </div>
                </div>
              )}
              {!isVoiceEditing &&
                activeReview &&
                openCorrection &&
                activeCorrectionSegment && (
                  <div className="correction-panel">
                    <div className="correction-panel-header">
                      <div>
                        <p className="correction-label">A DIFFERENT TAKE</p>
                        <p className="correction-hint">
                          Tap a version that sounds like what you said.
                        </p>
                      </div>
                      <button
                        className="text-button"
                        type="button"
                        onClick={() => setOpenCorrection(null)}
                        aria-label="Close word suggestions"
                      >
                        Done
                      </button>
                    </div>
                    <div className="correction-options">
                      {activeCorrectionSegment.alternatives
                        .filter(
                          (alternative) =>
                            alternative !== activeCorrectionSegment.selected,
                        )
                        .map((alternative) => (
                          <button
                            className="correction-option"
                            type="button"
                            key={alternative}
                            onClick={() =>
                              chooseAlternative(
                                openCorrection.segmentIndex,
                                alternative,
                              )
                            }
                          >
                            {alternative}
                          </button>
                        ))}
                    </div>
                  </div>
                )}
              {!isVoiceEditing && (
                <div className="note-actions">
                  {confirmingDelete ? (
                    <>
                      <span className="delete-prompt">
                        Delete this thought?
                      </span>
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
                          const updatedReviews = { ...transcriptReviews }
                          delete updatedReviews[activeParagraphIndex]
                          persistTranscriptReviews(updatedReviews)
                          setOpenCorrection(null)
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
                          setIsVoiceEditing(true)
                          setVoiceEditSelection(null)
                          setIsExtendingSelection(false)
                          setVoiceEditUndo(null)
                          setOpenCorrection(null)
                        }}
                      >
                        Edit by voice
                      </button>
                      <button
                        className="text-button text-button-subtle"
                        type="button"
                        onClick={() => {
                          setNoteDraft(activeNote)
                          setIsEditingNote(true)
                        }}
                      >
                        Edit with keyboard
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
              )}
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
                  const updatedReviews = { ...transcriptReviews }
                  delete updatedReviews[activeParagraphIndex]
                  persistTranscriptReviews(updatedReviews)
                  setOpenCorrection(null)
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
                  EDIT YOUR THOUGHT
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
              <p className="composer-guidance">
                Keep the words that still sound like you. Your note changes only
                when you save.
              </p>
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
                  Save changes <span aria-hidden="true">↗</span>
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
        <span>
          {activeParagraphIndex === null
            ? 'select a passage first'
            : isVoiceEditing
              ? voiceEditSelection
                ? 'hold to replace selected words'
                : 'tap a word to begin editing'
              : voiceState === 'recording'
                ? 'release to finish thought'
                : 'hold to speak'}
        </span>
      </div>

      {isSettingsOpen && (
        <div
          className="settings-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setIsSettingsOpen(false)
          }}
        >
          <section
            className="settings-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
          >
            <div className="settings-heading">
              <div>
                <p className="eyebrow">YOUR WORKSPACE</p>
                <h2 id="settings-title">Voice settings</h2>
              </div>
              <button
                className="settings-close"
                type="button"
                aria-label="Close settings"
                onClick={() => setIsSettingsOpen(false)}
              >
                ×
              </button>
            </div>
            <p className="settings-intro">
              Choose one of the speech models verified with your NVIDIA NIM key.
            </p>
            <form
              onSubmit={(event) => {
                event.preventDefault()
                try {
                  window.localStorage.setItem(
                    SPEECH_SETTINGS_KEY,
                    JSON.stringify(settingsDraft),
                  )
                  setSpeechSettings(settingsDraft)
                  setSettingsSaved(true)
                  setSettingsStorageError(false)
                } catch {
                  setSettingsSaved(false)
                  setSettingsStorageError(true)
                }
              }}
            >
              <p className="settings-provider">NVIDIA NIM</p>
              <label className="settings-label" htmlFor="speech-api-key">
                NVIDIA NIM API key
              </label>
              <input
                id="speech-api-key"
                type="password"
                autoComplete="off"
                spellCheck={false}
                placeholder="Paste your NVIDIA NIM API key"
                value={settingsDraft.apiKey}
                onChange={(event) => {
                  setSettingsDraft({
                    ...settingsDraft,
                    apiKey: event.target.value,
                  })
                  setSettingsSaved(false)
                }}
              />
              <label className="settings-label" htmlFor="speech-model">
                Speech model
              </label>
              <select
                id="speech-model"
                value={settingsDraft.modelId}
                onChange={(event) => {
                  setSettingsDraft({
                    ...settingsDraft,
                    modelId: event.target.value as SpeechModelId,
                  })
                  setSettingsSaved(false)
                }}
              >
                {speechModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} — {model.detail}
                  </option>
                ))}
              </select>
              <p className="settings-model-note">
                {
                  speechModels.find(
                    (model) => model.id === settingsDraft.modelId,
                  )?.mode
                }
              </p>
              <p className="settings-storage-note">
                The key is stored in this browser. Recorded audio is sent to
                NVIDIA for transcription.
              </p>
              <div className="settings-footer">
                {settingsStorageError ? (
                  <span className="settings-feedback error">
                    Could not save in browser storage.
                  </span>
                ) : settingsSaved ? (
                  <span className="settings-feedback">
                    Key saved on this device.
                  </span>
                ) : speechSettings.apiKey ? (
                  <button
                    className="text-button danger"
                    type="button"
                    onClick={() => {
                      try {
                        window.localStorage.removeItem(SPEECH_SETTINGS_KEY)
                        const cleared = { ...settingsDraft, apiKey: '' }
                        setSettingsDraft(cleared)
                        setSpeechSettings(cleared)
                        window.localStorage.setItem(
                          SPEECH_SETTINGS_KEY,
                          JSON.stringify(cleared),
                        )
                        setSettingsSaved(false)
                      } catch {
                        setSettingsStorageError(true)
                      }
                    }}
                  >
                    Remove saved key
                  </button>
                ) : (
                  <span />
                )}
                <button
                  className="save-note settings-save"
                  type="submit"
                  disabled={!settingsDraft.apiKey.trim()}
                >
                  Save settings <span aria-hidden="true">↗</span>
                </button>
              </div>
            </form>
          </section>
        </div>
      )}
    </div>
  )
}

export default App
