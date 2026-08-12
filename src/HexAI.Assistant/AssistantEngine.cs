using System.Text;
using System.Text.Json;
using HexAI.NLP;
using HexAI.NLP.Generative;
using HexAI.Assistant.Config;
using HexAI.Assistant.Knowledge;
using HexAI.Assistant.Skills;

namespace HexAI.Assistant;

/// <summary>
/// HexAI Asistan Motoru — tüm bileşenleri birleştiren ana orchestrator.
/// Kullanıcıdan mesaj alır, niyetini anlar, uygun skill'i çalıştırır veya
/// MiniGPT ile yanıt üretir.
/// </summary>
public class AssistantEngine
{
    private readonly AssistantConfig _config;
    private readonly NeuralIntentClassifier _intentClassifier;
    private readonly ConversationMemory _memory;
    private readonly Dictionary<IntentClassifier.Intent, ISkill> _skills;
    private LargeGPT? _model;
    private bool _modelLoaded = false;

    /// <summary>Model eğitilmiş ve hazır mı?</summary>
    public bool IsModelReady => _modelLoaded;

    /// <summary>Yapılandırma ayarları.</summary>
    public AssistantConfig Config => _config;

    /// <summary>Sohbet geçmişi.</summary>
    public ConversationMemory Memory => _memory;

    /// <summary>
    /// Yeni bir asistan motoru oluşturur.
    /// </summary>
    public AssistantEngine(AssistantConfig? config = null)
    {
        _config = config ?? AssistantConfig.Load();
        _config.EnsureDirectories();

        _intentClassifier = new NeuralIntentClassifier();
        _intentClassifier.TrainSynthetic(epochs: 100); // Hızlı eğitim

        _memory = new ConversationMemory(_config.MemorySize);

        // Skill'leri kaydet
        _skills = new Dictionary<IntentClassifier.Intent, ISkill>
        {
            [IntentClassifier.Intent.FileOperation] = new FileSystemSkill(),
            [IntentClassifier.Intent.RunCommand] = new PowerShellSkill(),
            [IntentClassifier.Intent.SystemInfo] = new SystemInfoSkill(),
            [IntentClassifier.Intent.Math] = new MathSkill(),
            [IntentClassifier.Intent.OpenApplication] = new ProcessSkill(),
        };
    }

    /// <summary>
    /// Başlangıçta son modeli otomatik yükler.
    /// </summary>
    public void Initialize()
    {
        // Son kullanılan modeli yüklemeyi dene
        if (!string.IsNullOrWhiteSpace(_config.LastModelName))
        {
            try
            {
                LoadModel(_config.LastModelName);
            }
            catch { /* Model yüklenemezse sessizce geç */ }
        }

        // Varsayılan corpus varsa ve model yoksa eğit
        if (!_modelLoaded)
        {
            string defaultCorpus = Path.Combine(_config.GetCorpusPath(), "default_turkish.txt");
            if (File.Exists(defaultCorpus))
            {
                try
                {
                    TrainFromFile(defaultCorpus, _config.DefaultEpochs, silent: true);
                }
                catch { /* Eğitim başarısızsa geç */ }
            }
        }
    }

    /// <summary>
    /// Kullanıcı mesajını işler ve yanıt döndürür.
    /// Bu, asistanın ana giriş noktasıdır.
    /// </summary>
    public async Task<string> ProcessMessageAsync(string userMessage)
    {
        if (string.IsNullOrWhiteSpace(userMessage))
            return "Bir şey söylemedim gibi geldi. Nasıl yardımcı olabilirim?";

        // Mesajı hafızaya ekle
        _memory.AddUserMessage(userMessage);

        // Niyeti algıla
        var intentResult = _intentClassifier.Classify(userMessage);

        // Niyete göre yanıt üret
        string response = intentResult.Intent switch
        {
            IntentClassifier.Intent.Chat => HandleChat(userMessage, intentResult),
            IntentClassifier.Intent.Help => GetHelpText(),
            IntentClassifier.Intent.Train => await HandleTrainAsync(intentResult),
            IntentClassifier.Intent.ModelManage => HandleModelManage(intentResult),
            IntentClassifier.Intent.FileOperation or
            IntentClassifier.Intent.RunCommand or
            IntentClassifier.Intent.SystemInfo or
            IntentClassifier.Intent.Math or
            IntentClassifier.Intent.OpenApplication => await HandleSkillAsync(intentResult),
            _ => HandleChat(userMessage, intentResult)
        };

        // Yanıtı hafızaya ekle
        _memory.AddAssistantMessage(response);

        return response;
    }

    #region Chat İşleme

    private string HandleChat(string input, IntentClassifier.IntentResult intentResult)
    {
        // Özel chat komutları
        if (intentResult.Parameters.ContainsKey("action"))
        {
            return intentResult.Parameters["action"] switch
            {
                "clear" => ClearHistory(),
                "history" => GetHistory(),
                "exit" => "exit",
                _ => ""
            };
        }

        string lower = input.ToLowerInvariant();

        // Selamlama kalıpları
        if (ContainsAny(lower, "merhaba", "selam", "hey", "sa", "selamünaleyküm"))
            return $"Merhaba! 👋 Ben {_config.AssistantName}, senin kişisel yapay zeka asistanın. Sana nasıl yardımcı olabilirim?";

        if (ContainsAny(lower, "nasılsın", "naber", "ne haber", "iyi misin"))
            return "Teşekkürler, gayet iyiyim! 😊 Sana yardımcı olmaya hazırım. Ne yapmamı istersin?";

        if (ContainsAny(lower, "günaydın"))
            return "Günaydın! ☀️ Bugün sana nasıl yardımcı olabilirim?";

        if (ContainsAny(lower, "iyi akşamlar", "iyi geceler"))
            return "İyi akşamlar! 🌙 Başka bir isteğin var mı?";

        if (ContainsAny(lower, "teşekkür", "sağol", "sağ ol", "eyvallah"))
            return "Rica ederim! 😊 Başka bir konuda yardıma ihtiyacın olursa buradayım.";

        if (ContainsAny(lower, "hoşçakal", "görüşürüz", "bay bay", "bb"))
            return "Hoşça kal! 👋 Tekrar görüşmek üzere!";

        if (ContainsAny(lower, "sen kimsin", "adın ne", "nesin sen", "kendini tanıt"))
            return $"Ben {_config.AssistantName}! 🤖 Tamamen C# ile sıfırdan yazılmış bir yapay zeka asistanıyım.\n" +
                   $"İçimde kendi Transformer motorumu, sinir ağımı ve NLP altyapımı barındırıyorum.\n" +
                   $"Dosya işlemleri, PowerShell komutları, sistem bilgisi sorgulama ve daha fazlasını yapabilirim.\n" +
                   $"Beni eğitebilir ve zamanla daha da akıllı hale getirebilirsin!";

        if (ContainsAny(lower, "ne yapabilirsin", "yeteneklerin", "neler yaparsın", "özellikler"))
            return GetHelpText();

        // Model varsa MiniGPT ile yanıt üretmeyi dene
        if (_modelLoaded && _model != null)
        {
            try
            {
                string generated = _model.Generate(input, generateLength: 8, temperature: _config.Temperature);
                if (!string.IsNullOrWhiteSpace(generated) && generated.Length > input.Length + 3)
                {
                    return $"💭 {generated}";
                }
            }
            catch { /* MiniGPT üretim hatası — fallback'e geç */ }
        }

        // Fallback yanıtlar
        return GetFallbackResponse(input);
    }

    private string GetFallbackResponse(string input)
    {
        var fallbacks = new[]
        {
            "Bu konuda henüz yeterli bilgim yok. Beni daha fazla veriyle eğitirsen daha iyi yanıtlar verebilirim! " +
            "(`/train <dosya_yolu>` komutuyla eğitim yapabilirsin)",
            "Hmm, bu konuyu tam anlayamadım. 🤔 Belki şu yeteneklerimden biri işine yarar:\n" +
            "• Dosya işlemleri: \"masaüstündeki dosyaları listele\"\n" +
            "• Sistem bilgisi: \"disk alanım ne kadar?\"\n" +
            "• PowerShell: \"/ps Get-Process\"\n" +
            "• Matematik: \"25 çarpı 16 kaç eder?\"\n" +
            "`/help` yazarak tüm yeteneklerimi görebilirsin.",
            "Bu konuda daha fazla bilgiye ihtiyacım var. Beni eğitmek için `/train` komutunu kullanabilirsin!",
        };

        var rng = new Random();
        return fallbacks[rng.Next(fallbacks.Length)];
    }

    #endregion

    #region Skill İşleme

    private async Task<string> HandleSkillAsync(IntentClassifier.IntentResult intentResult)
    {
        if (_skills.TryGetValue(intentResult.Intent, out var skill))
        {
            var result = await skill.ExecuteAsync(intentResult.OriginalInput, intentResult.Parameters);
            return result.Response;
        }

        return "Bu yetenek henüz aktif değil.";
    }

    #endregion

    #region Eğitim İşleme

    private async Task<string> HandleTrainAsync(IntentClassifier.IntentResult intentResult)
    {
        string path = intentResult.Parameters.GetValueOrDefault("path", "");
        string command = intentResult.Parameters.GetValueOrDefault("command", "/train");

        if (string.IsNullOrWhiteSpace(path))
            return "⚠️ Lütfen eğitim dosyası veya klasör yolunu belirt.\n" +
                   "Örnek: `/train C:\\veriler\\metin.txt` veya `/train-dir C:\\veriler\\`";

        try
        {
            if (command == "/train-dir")
            {
                return TrainFromDirectory(path, _config.DefaultEpochs);
            }
            else
            {
                return TrainFromFile(path, _config.DefaultEpochs);
            }
        }
        catch (Exception ex)
        {
            return $"❌ Eğitim hatası: {ex.Message}";
        }
    }

    /// <summary>
    /// Tek dosyadan model eğitimi.
    /// </summary>
    public string TrainFromFile(string filePath, int epochs, bool silent = false)
    {
        string corpus = CorpusLoader.LoadFromFile(filePath);
        return TrainFromCorpus(corpus, Path.GetFileName(filePath), epochs, silent);
    }

    /// <summary>
    /// Klasördeki tüm dosyalardan model eğitimi.
    /// </summary>
    public string TrainFromDirectory(string directoryPath, int epochs)
    {
        string corpus = CorpusLoader.LoadFromDirectory(directoryPath);
        return TrainFromCorpus(corpus, Path.GetFileName(directoryPath), epochs);
    }

    /// <summary>
    /// Ham corpus metni üzerinden model eğitimi.
    /// </summary>
    public string TrainFromCorpus(string corpus, string sourceName, int epochs, bool silent = false)
    {
        var info = CorpusLoader.GetInfo(corpus);

        if (!silent)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"\n📚 Eğitim kaynağı: {sourceName}");
            Console.WriteLine(info.ToString());
            Console.ResetColor();
        }

        _model ??= new LargeGPT(
            hiddenSize: _config.HiddenSize, 
            numHeads: _config.NumHeads,
            numLayers: _config.NumLayers,
            sequenceLength: _config.SequenceLength,
            learningRate: _config.LearningRate, 
            seed: 42);

        int reportInterval = silent ? int.MaxValue : Math.Max(1, epochs / 10);

        _model.Train(corpus, epochs, onEpochComplete: silent ? null : (epoch, loss) =>
        {
            // Raporlama Network.Train içinden yapılıyor
        });

        _modelLoaded = true;

        if (_config.AutoSave)
        {
            SaveModel("autosave");
        }

        return $"✅ Eğitim tamamlandı!\n{info}\n💾 Model otomatik kaydedildi.";
    }

    #endregion

    #region Model Yönetimi

    private string HandleModelManage(IntentClassifier.IntentResult intentResult)
    {
        string action = intentResult.Parameters.GetValueOrDefault("action", "status");

        return action switch
        {
            "save" => SaveModel(intentResult.Parameters.GetValueOrDefault("name", "default")),
            "load" => LoadModel(intentResult.Parameters.GetValueOrDefault("name", "default")),
            "list" => ListModels(),
            "status" => GetModelStatus(),
            _ => GetModelStatus()
        };
    }

    /// <summary>
    /// Modeli diske kaydeder.
    /// </summary>
    public string SaveModel(string name)
    {
        if (_model == null || !_modelLoaded)
            return "⚠️ Kaydedilecek model yok. Önce eğitim yapın.";

        string modelDir = Path.Combine(_config.GetModelsPath(), name);
        Directory.CreateDirectory(modelDir);

        try
        {
            // Model metadatasını kaydet
            var metadata = new ModelMetadata
            {
                Name = name,
                SaveDate = DateTime.Now,
                HiddenSize = _config.HiddenSize,
                LearningRate = _config.LearningRate,
            };
            var metaJson = JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(Path.Combine(modelDir, "metadata.json"), metaJson);

            // Config'e son model adını kaydet
            _config.LastModelName = name;
            _config.Save();

            return $"✅ Model '{name}' kaydedildi: {modelDir}";
        }
        catch (Exception ex)
        {
            return $"❌ Model kaydetme hatası: {ex.Message}";
        }
    }

    /// <summary>
    /// Modeli diskten yükler.
    /// </summary>
    public string LoadModel(string name)
    {
        string modelDir = Path.Combine(_config.GetModelsPath(), name);

        if (!Directory.Exists(modelDir))
            return $"⚠️ Model bulunamadı: {name}";

        try
        {
            string metaPath = Path.Combine(modelDir, "metadata.json");
            if (File.Exists(metaPath))
            {
                var metaJson = File.ReadAllText(metaPath);
                var metadata = JsonSerializer.Deserialize<ModelMetadata>(metaJson);
                if (metadata != null)
                {
                    _config.LastModelName = name;
                    _config.Save();

                    // Corpus'tan yeniden eğit
                    string corpusPath = Path.Combine(_config.GetCorpusPath(), "default_turkish.txt");
                    if (File.Exists(corpusPath))
                    {
                        TrainFromFile(corpusPath, _config.DefaultEpochs, silent: true);
                    }

                    return $"✅ Model '{name}' yüklendi.\n" +
                           $"  Kayıt: {metadata.SaveDate:dd/MM/yyyy HH:mm}\n" +
                           $"  Gizli Boyut: {metadata.HiddenSize}";
                }
            }

            _modelLoaded = true;
            return $"✅ Model '{name}' yüklendi.";
        }
        catch (Exception ex)
        {
            return $"❌ Model yükleme hatası: {ex.Message}";
        }
    }

    /// <summary>
    /// Kaydedilmiş modelleri listeler.
    /// </summary>
    public string ListModels()
    {
        string modelsPath = _config.GetModelsPath();
        if (!Directory.Exists(modelsPath))
            return "📁 Henüz kaydedilmiş model yok.";

        var dirs = Directory.GetDirectories(modelsPath);
        if (dirs.Length == 0)
            return "📁 Henüz kaydedilmiş model yok.";

        var sb = new StringBuilder();
        sb.AppendLine("📁 Kaydedilmiş Modeller:\n");

        foreach (var dir in dirs)
        {
            string name = Path.GetFileName(dir);
            string metaPath = Path.Combine(dir, "metadata.json");
            string dateStr = "?";

            if (File.Exists(metaPath))
            {
                try
                {
                    var meta = JsonSerializer.Deserialize<ModelMetadata>(File.ReadAllText(metaPath));
                    if (meta != null) dateStr = meta.SaveDate.ToString("dd/MM/yyyy HH:mm");
                }
                catch { }
            }

            string active = name == _config.LastModelName ? " ← aktif" : "";
            sb.AppendLine($"  📦 {name} ({dateStr}){active}");
        }

        return sb.ToString().TrimEnd();
    }

    /// <summary>
    /// Model durumunu döndürür.
    /// </summary>
    public string GetModelStatus()
    {
        var sb = new StringBuilder();
        sb.AppendLine("📊 Model Durumu:\n");
        sb.AppendLine($"  Durum       : {(_modelLoaded ? "✅ Hazır" : "⚠️ Eğitilmemiş")}");
        sb.AppendLine($"  Gizli Boyut : {_config.HiddenSize}");
        sb.AppendLine($"  Attention K.: {_config.NumHeads}");
        sb.AppendLine($"  Transf. Kat.: {_config.NumLayers}");
        sb.AppendLine($"  Seq. Uzunluk: {_config.SequenceLength}");
        sb.AppendLine($"  Öğr. Oranı  : {_config.LearningRate}");
        sb.AppendLine($"  Sıcaklık    : {_config.Temperature}");
        sb.AppendLine($"  Son Model   : {_config.LastModelName ?? "(yok)"}");
        sb.AppendLine($"  Sohbet Sayısı: {_memory.Count}");

        return sb.ToString().TrimEnd();
    }

    #endregion

    #region Yardım

    /// <summary>
    /// Yardım metnini döndürür.
    /// </summary>
    public string GetHelpText()
    {
        return @$"
🤖 {_config.AssistantName} — Kişisel Yapay Zeka Asistanın

💬 Doğal Dil ile:
  • ""masaüstündeki dosyaları listele""
  • ""disk alanım ne kadar?""
  • ""saat kaç?""
  • ""25 çarpı 16 kaç eder?""
  • ""not defteri aç""
  • ""çalışan süreçleri göster""

⚡ Özel Komutlar:
  /help             Yardım menüsü
  /train <dosya>    Dosyadan eğitim
  /train-dir <kls>  Klasörden eğitim
  /save [isim]      Modeli kaydet
  /load [isim]      Modeli yükle
  /models           Modelleri listele
  /status           Model durumu
  /ps <komut>       PowerShell komutu
  /files [yol]      Dosyaları listele
  /read <dosya>     Dosya oku
  /clear            Sohbet geçmişini temizle
  /history          Geçmişi göster
  /exit             Çıkış";
    }

    #endregion

    #region Yardımcılar

    private string ClearHistory()
    {
        _memory.Clear();
        return "🗑️ Sohbet geçmişi temizlendi.";
    }

    private string GetHistory()
    {
        var messages = _memory.GetRecentMessages(20);
        if (messages.Count == 0)
            return "📜 Sohbet geçmişi boş.";

        var sb = new StringBuilder();
        sb.AppendLine("📜 Sohbet Geçmişi:\n");
        foreach (var msg in messages)
        {
            string icon = msg.Role == "user" ? "👤" : "🤖";
            sb.AppendLine($"  {icon} [{msg.Timestamp:HH:mm}] {msg.Content}");
        }
        return sb.ToString().TrimEnd();
    }

    private static bool ContainsAny(string text, params string[] keywords)
    {
        return keywords.Any(k => text.Contains(k));
    }

    #endregion
}

/// <summary>
/// Model metadata — kayıt bilgisi.
/// </summary>
public class ModelMetadata
{
    public string Name { get; set; } = "";
    public DateTime SaveDate { get; set; }
    public int HiddenSize { get; set; }
    public double LearningRate { get; set; }
}
