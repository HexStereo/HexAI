using System.Speech.Recognition;
using System.Speech.Synthesis;

namespace HexAI.CLI;

/// <summary>
/// Sistem sesini ve mikrofonu kontrol eden ses yöneticisi (STT ve TTS).
/// Sadece Windows platformunda System.Speech kütüphanesi üzerinden çalışır.
/// </summary>
public class VoiceManager : IDisposable
{
    private SpeechSynthesizer? _synthesizer;
    private SpeechRecognitionEngine? _recognizer;
    private bool _isListening;

    // Dinleme işlemi sırasında sonucu döndürebilmek için
    private TaskCompletionSource<string>? _tcs;

    public bool IsVoiceSupported { get; private set; }
    
    public VoiceManager()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                // TTS Kurulumu
                _synthesizer = new SpeechSynthesizer();
                _synthesizer.SetOutputToDefaultAudioDevice();
                
                // STT Kurulumu
                _recognizer = new SpeechRecognitionEngine();
                _recognizer.SetInputToDefaultAudioDevice();
                
                // Genel konuşma (dictation) grameri yükle
                _recognizer.LoadGrammar(new DictationGrammar());
                
                _recognizer.SpeechRecognized += Recognizer_SpeechRecognized;
                _recognizer.SpeechRecognitionRejected += Recognizer_SpeechRecognitionRejected;

                IsVoiceSupported = true;
            }
            else
            {
                IsVoiceSupported = false;
            }
        }
        catch (Exception)
        {
            // Mikrofon yoksa veya erişim izni yoksa kapat
            IsVoiceSupported = false;
        }
    }

    /// <summary>
    /// Verilen metni asistan sesiyle okur (TTS).
    /// </summary>
    public void Speak(string text)
    {
        if (!IsVoiceSupported || _synthesizer == null || string.IsNullOrWhiteSpace(text)) return;

        // Emojileri ve markdown işaretlerini temizle
        string cleanText = CleanForSpeech(text);
        if (!string.IsNullOrWhiteSpace(cleanText))
        {
            // Asenkron okur ki terminal bloke olmasın (fakat CLI thread'ini bekletmemiz gerekmez)
            _synthesizer.SpeakAsync(cleanText);
        }
    }

    /// <summary>
    /// Konuşmayı durdurur.
    /// </summary>
    public void StopSpeaking()
    {
        if (IsVoiceSupported && _synthesizer != null)
        {
            _synthesizer.SpeakAsyncCancelAll();
        }
    }

    /// <summary>
    /// Mikrofonu dinleyerek (STT) kullanıcının söylediğini metin olarak döndürür.
    /// </summary>
    public async Task<string> ListenAsync()
    {
        if (!IsVoiceSupported || _recognizer == null)
            return "";

        if (_isListening)
            return ""; // Zaten dinliyor

        try
        {
            _isListening = true;
            _tcs = new TaskCompletionSource<string>();

            // Tek seferlik dinleme başlat
            _recognizer.RecognizeAsync(RecognizeMode.Single);

            // Sonuç gelene kadar bekle (veya iptal edilene kadar)
            string result = await _tcs.Task;
            return result;
        }
        catch
        {
            return "";
        }
        finally
        {
            _isListening = false;
        }
    }

    /// <summary>
    /// Dinleme işlemini manuel olarak iptal eder.
    /// </summary>
    public void StopListening()
    {
        if (IsVoiceSupported && _recognizer != null && _isListening)
        {
            _recognizer.RecognizeAsyncCancel();
            _tcs?.TrySetResult("");
        }
    }

    private void Recognizer_SpeechRecognized(object? sender, SpeechRecognizedEventArgs e)
    {
        if (e.Result.Confidence > 0.5f) // Güven eşiği
        {
            _tcs?.TrySetResult(e.Result.Text);
        }
        else
        {
            _tcs?.TrySetResult(""); // Anlaşılamadı
        }
    }

    private void Recognizer_SpeechRecognitionRejected(object? sender, SpeechRecognitionRejectedEventArgs e)
    {
        _tcs?.TrySetResult(""); // Anlaşılamadı
    }

    /// <summary>
    /// Sentezleyiciye (TTS) gönderilecek metni temizler.
    /// Emojiler ve kod blokları okunmaz.
    /// </summary>
    private static string CleanForSpeech(string input)
    {
        // Kod bloklarını (```) kaldır
        input = System.Text.RegularExpressions.Regex.Replace(input, @"```[\s\S]*?```", " [kod bloğu] ");
        // Markdown formatlamaları
        input = input.Replace("**", "").Replace("*", "");
        // Temel Emojiler (Hepsini temizlemek regex ile mümkün ama şimdilik spesifik olanları kaldıralım)
        string[] emojis = { "👋", "😊", "🤖", "☀️", "🌙", "📁", "📄", "💾", "🧠", "⚙️", "🔄", "🌐", "💻", "🔍", "✅", "⚠️", "❌", "⏳", "🗑️", "📜" };
        foreach (var emoji in emojis)
        {
            input = input.Replace(emoji, "");
        }
        
        return input.Trim();
    }

    public void Dispose()
    {
        _synthesizer?.Dispose();
        _recognizer?.Dispose();
    }
}
