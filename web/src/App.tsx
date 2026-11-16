import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

const NOTES_STORAGE_KEY = 'margin:sample-essay:notes'
const SPEECH_SETTINGS_KEY = 'margin:speech-settings'

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

  const selectParagraph = (index: number) => {
    setActiveParagraphIndex(index)
    setNoteDraft(notes[index] ?? '')
    setIsEditingNote(false)
    setConfirmingDelete(false)
  }

  const startVoiceCapture = useCallback(async () => {
    if (recorderRef.current || !voiceHoldRef.current) return
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
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        throw new Error('This browser does not support microphone recording.')
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
          }
          if (result.type === 'transcript' && result.text) {
            capture.latestText = result.text
            setNoteDraft(
              appendTranscript(notes[paragraphIndex] ?? '', result.text),
            )
          } else if (result.type === 'done') {
            const transcript = (result.text || capture.latestText).trim()
            if (transcript) {
              const updatedNote = appendTranscript(
                notes[paragraphIndex] ?? '',
                transcript,
              )
              persistNotes({ ...notes, [paragraphIndex]: updatedNote })
              setNoteDraft(updatedNote)
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
              },
              body: wav,
            })
            const result: { text?: string; detail?: string } =
              await response.json()
            if (!response.ok)
              throw new Error(result.detail ?? 'Transcription failed.')
            const transcript = result.text?.trim()
            if (!transcript)
              throw new Error(
                'No speech was detected. Try speaking a little longer.',
              )

            const existingNote = notes[paragraphIndex] ?? ''
            const updatedNote = appendTranscript(existingNote, transcript)
            persistNotes({ ...notes, [paragraphIndex]: updatedNote })
            setNoteDraft(updatedNote)
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
  }, [activeParagraphIndex, notes, persistNotes, speechSettings])

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
          'input, textarea, select, [contenteditable="true"], button:not(.voice-capture)',
        )
      )
        return
      event.preventDefault()
      if (
        event.repeat ||
        isEditingNote ||
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
    isSettingsOpen,
    startVoiceCapture,
    stopVoiceCapture,
    voiceState,
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
              disabled={voiceState === 'transcribing' || isEditingNote}
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
                ? 'Release to finish thought'
                : voiceState === 'transcribing'
                  ? 'Transcribing…'
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
                ? 'Listening — release to finish your thought.'
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
                <p className="saved-note-label">YOUR THOUGHT</p>
                <span className="saved-note-status">WRITING</span>
              </div>
              <p className="saved-note-text">{noteDraft}</p>
            </div>
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
        <span>
          {activeParagraphIndex === null
            ? 'select a passage first'
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
