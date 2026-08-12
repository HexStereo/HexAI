using System.Text.Json;

namespace HexAI.Assistant.Config;

/// <summary>
/// Asistan yapılandırma ayarları.
/// JSON dosyasından okunur ve yazılır.
/// </summary>
public class AssistantConfig
{
    /// <summary>Asistanın adı.</summary>
    public string AssistantName { get; set; } = "HexAI";

    /// <summary>Kullanıcı adı.</summary>
    public string UserName { get; set; } = "Kullanıcı";

    /// <summary>Varsayılan model dizini.</summary>
    public string ModelsDirectory { get; set; } = "data/models";

    /// <summary>Eğitim verisi dizini.</summary>
    public string CorpusDirectory { get; set; } = "data/corpus";

    /// <summary>Log dizini.</summary>
    public string LogsDirectory { get; set; } = "data/logs";

    /// <summary>MiniGPT gizli katman boyutu.</summary>
    public int HiddenSize { get; set; } = 64;

    /// <summary>Attention kafa sayısı (LargeGPT için).</summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>Transformer blok sayısı (LargeGPT için).</summary>
    public int NumLayers { get; set; } = 2;

    /// <summary>Sıra uzunluğu (LargeGPT için bağlam penceresi).</summary>
    public int SequenceLength { get; set; } = 16;

    /// <summary>Öğrenme oranı.</summary>
    public double LearningRate { get; set; } = 0.05;

    /// <summary>Varsayılan eğitim epoch sayısı.</summary>
    public int DefaultEpochs { get; set; } = 500;

    /// <summary>Metin üretim sıcaklığı (temperature).</summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>Sohbet hafıza boyutu (son N mesaj).</summary>
    public int MemorySize { get; set; } = 50;

    /// <summary>Son aktif model adı.</summary>
    public string? LastModelName { get; set; }

    /// <summary>Otomatik kaydetme açık mı?</summary>
    public bool AutoSave { get; set; } = true;

    /// <summary>Yazma animasyonu açık mı?</summary>
    public bool TypewriterEffect { get; set; } = true;

    /// <summary>Yazma animasyonu gecikme (ms).</summary>
    public int TypewriterDelayMs { get; set; } = 15;

    /// <summary>Sesli yanıt (Text-to-Speech) açık mı?</summary>
    public bool VoiceOutputEnabled { get; set; } = true;

    /// <summary>Yapılandırma dosyası yolu.</summary>
    private static string ConfigPath => Path.Combine(
        AppDomain.CurrentDomain.BaseDirectory, "data", "config.json");

    /// <summary>
    /// Yapılandırmayı dosyadan yükler. Yoksa varsayılan oluşturur.
    /// </summary>
    public static AssistantConfig Load()
    {
        try
        {
            if (File.Exists(ConfigPath))
            {
                var json = File.ReadAllText(ConfigPath);
                return JsonSerializer.Deserialize<AssistantConfig>(json) ?? new AssistantConfig();
            }
        }
        catch { /* Sessizce varsayılanı döndür */ }

        var config = new AssistantConfig();
        config.Save(); // Varsayılan dosyayı oluştur
        return config;
    }

    /// <summary>
    /// Yapılandırmayı dosyaya kaydeder.
    /// </summary>
    public void Save()
    {
        try
        {
            var dir = Path.GetDirectoryName(ConfigPath);
            if (dir != null) Directory.CreateDirectory(dir);

            var json = JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(ConfigPath, json);
        }
        catch { /* Kaydetme hatası sessizce geçilir */ }
    }

    /// <summary>
    /// Gerekli dizinleri oluşturur.
    /// </summary>
    public void EnsureDirectories()
    {
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        Directory.CreateDirectory(Path.Combine(baseDir, ModelsDirectory));
        Directory.CreateDirectory(Path.Combine(baseDir, CorpusDirectory));
        Directory.CreateDirectory(Path.Combine(baseDir, LogsDirectory));
    }

    /// <summary>Tam model dizin yolunu döndürür.</summary>
    public string GetModelsPath() =>
        Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ModelsDirectory);

    /// <summary>Tam corpus dizin yolunu döndürür.</summary>
    public string GetCorpusPath() =>
        Path.Combine(AppDomain.CurrentDomain.BaseDirectory, CorpusDirectory);

    /// <summary>Tam log dizin yolunu döndürür.</summary>
    public string GetLogsPath() =>
        Path.Combine(AppDomain.CurrentDomain.BaseDirectory, LogsDirectory);
}
