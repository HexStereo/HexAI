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
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.InputEncoding = System.Text.Encoding.UTF8;

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
        ConsoleUI.PrintLoading("Asistan ve Yapay Zeka Motoru başlatılıyor...");
        var engine = new AssistantEngine(config);

        // Araştırma / araç durum güncellemelerini ekrana yaz
        engine.OnStatus += status =>
        {
            Console.ForegroundColor = ConsoleColor.DarkCyan;
            Console.WriteLine($"  {status}");
            Console.ResetColor();
        };

        // Motoru başlat (Öz Transformer, NLU, Bilgi Tabanı ve Araştırma Motoru)
        engine.Initialize();
        ConsoleUI.ClearLoading();

        // Yerel model durumu
        string modelName = config.LastModelName ?? "LargeGPT (Öz Transformer)";
        ConsoleUI.PrintStatus(engine.IsModelReady, modelName);

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

            string trimmed = input.Trim().ToLower(new System.Globalization.CultureInfo("tr-TR"));

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

                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write("  🤖 ");

                var responseBuffer = new System.Text.StringBuilder();

                await foreach (var token in engine.ProcessMessageStreamAsync(input.Trim()))
                {
                    responseBuffer.Append(token);
                    Console.Write(token);
                }

                Console.ResetColor();
                Console.WriteLine();
                Console.WriteLine();

                string fullResponse = responseBuffer.ToString().Trim();

                // "exit" özel sinyali
                if (fullResponse == "exit")
                {
                    ConsoleUI.PrintResponse("Hoşça kal! 👋 Tekrar görüşmek üzere!", false, 0);
                    if (config.VoiceOutputEnabled) voiceManager.Speak("Hoşça kal, tekrar görüşmek üzere!");
                    break;
                }

                if (config.VoiceOutputEnabled && !string.IsNullOrWhiteSpace(fullResponse))
                {
                    voiceManager.Speak(fullResponse);
                }
            }
            catch (Exception ex)
            {
                ConsoleUI.PrintError($"Beklenmeyen hata: {ex.Message}");
            }
        }
    }
}

