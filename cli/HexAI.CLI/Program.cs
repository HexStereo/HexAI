using HexAI.Assistant;
using HexAI.Assistant.Config;

namespace HexAI.CLI;

/// <summary>
/// HexAI Kişisel Yapay Zeka Asistanı — Ana Giriş Noktası.
/// PowerShell'den çalıştırılır ve etkileşimli sohbet döngüsü başlatır.
/// 
/// Kullanım:
///   dotnet run --project cli/HexAI.CLI
/// </summary>
class Program
{
    static async Task Main(string[] args)
    {
        // Yapılandırmayı yükle
        var config = AssistantConfig.Load();

        // Banner ve hoşgeldin mesajı
        ConsoleUI.PrintBanner();
        ConsoleUI.PrintWelcome();

        // Ses yöneticisini başlat
        using var voiceManager = new VoiceManager();
        bool voiceMode = false;

        if (voiceManager.IsVoiceSupported)
        {
            ConsoleUI.PrintInfo("🎤 Ses desteği hazır. Sesli konuşmak için '/voice' yazabilirsiniz.");
        }

        // Asistan motorunu başlat
        ConsoleUI.PrintLoading("Asistan başlatılıyor...");
        var engine = new AssistantEngine(config);

        // Varsayılan corpus varsa ve model eğitilmemişse, sessizce eğit
        string corpusPath = Path.Combine(config.GetCorpusPath(), "default_turkish.txt");
        if (File.Exists(corpusPath) && !engine.IsModelReady)
        {
            ConsoleUI.ClearLoading();
            ConsoleUI.PrintInfo("Varsayılan bilgi kütüphanesi yükleniyor...");
            try
            {
                engine.TrainFromFile(corpusPath, config.DefaultEpochs, silent: true);
                ConsoleUI.PrintSuccess("Bilgi kütüphanesi yüklendi!");
            }
            catch (Exception ex)
            {
                ConsoleUI.PrintError($"Bilgi kütüphanesi yüklenemedi: {ex.Message}");
            }
        }
        else
        {
            ConsoleUI.ClearLoading();
        }

        // Durum bilgisi
        ConsoleUI.PrintStatus(engine.IsModelReady, config.LastModelName);

        // === ANA SOHBET DÖNGÜSÜ (REPL) ===
        while (true)
        {
            // Prompt göster
            if (voiceMode)
            {
                ConsoleUI.PrintInfo("🎤 Dinliyorum... (Çıkmak için 'kapat' deyin veya konsola bir şey yazıp Enter'a basın)");
            }
            else
            {
                ConsoleUI.PrintPrompt(config.AssistantName);
            }

            // Kullanıcı girdisini oku veya dinle
            string? input = "";
            if (voiceMode)
            {
                // Asenkron dinlerken klavye kesintisine izin vermek için
                var listenTask = voiceManager.ListenAsync();
                var readTask = Task.Run(() => Console.ReadLine());

                var completedTask = await Task.WhenAny(listenTask, readTask);
                
                if (completedTask == listenTask)
                {
                    input = await listenTask;
                    if (string.IsNullOrWhiteSpace(input)) continue;
                    Console.WriteLine($"\n🗣️ Siz: {input}");
                }
                else
                {
                    input = await readTask;
                    voiceManager.StopListening();
                }
            }
            else
            {
                input = Console.ReadLine();
            }

            if (string.IsNullOrWhiteSpace(input))
                continue;

            string trimmed = input.Trim().ToLowerInvariant();

            // Ses modunu aç/kapat
            if (trimmed == "/voice")
            {
                if (!voiceManager.IsVoiceSupported)
                {
                    ConsoleUI.PrintError("Ses desteği (System.Speech) sisteminizde kullanılamıyor.");
                    continue;
                }
                
                voiceMode = !voiceMode;
                string state = voiceMode ? "açıldı" : "kapatıldı";
                ConsoleUI.PrintSuccess($"Sesli asistan modu {state}.");
                continue;
            }

            // Çıkış kontrolü
            if (trimmed is "/exit" or "/quit" or "/q" or "çıkış" or "çık" or "exit" or "quit" or "kapat")
            {
                ConsoleUI.PrintResponse("Hoşça kal! 👋 Tekrar görüşmek üzere!", 
                    config.TypewriterEffect, config.TypewriterDelayMs);
                if (config.VoiceOutputEnabled) voiceManager.Speak("Hoşça kal, tekrar görüşmek üzere!");
                break;
            }

            try
            {
                // Mevcut konuşmayı durdur
                voiceManager.StopSpeaking();
                // Mesajı asistan motoruna gönder
                string response = await engine.ProcessMessageAsync(input.Trim());

                // "exit" özel sinyali
                if (response == "exit")
                {
                    ConsoleUI.PrintResponse("Hoşça kal! 👋 Tekrar görüşmek üzere!",
                        config.TypewriterEffect, config.TypewriterDelayMs);
                    if (config.VoiceOutputEnabled) voiceManager.Speak("Hoşça kal, tekrar görüşmek üzere!");
                    break;
                }

                // Yanıtı göster ve oku
                ConsoleUI.PrintResponse(response, config.TypewriterEffect, config.TypewriterDelayMs);
                
                if (config.VoiceOutputEnabled)
                {
                    voiceManager.Speak(response);
                }
            }
            catch (Exception ex)
            {
                ConsoleUI.PrintError($"Beklenmeyen hata: {ex.Message}");
            }
        }
    }
}
