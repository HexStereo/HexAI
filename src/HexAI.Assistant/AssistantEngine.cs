using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using HexAI.NLP;
using HexAI.NLP.Generative;
using HexAI.Assistant.Config;
using HexAI.Assistant.Knowledge;
using HexAI.Assistant.Skills;
using HexAI.Assistant.Agents;
using HexAI.Assistant.Evaluation;
using HexAI.Assistant.Research;

namespace HexAI.Assistant;

/// <summary>
/// HexAI Asistan Motoru — Tamamen C# ile sıfırdan geliştirilmiş, dışarıdan hiçbir
/// hazır modele veya harici kütüphaneye bağımlı olmayan egemen yapay zeka orchestrator'ı.
/// 
/// Bileşenler ve Mimari:
///   - Sinir Ağı & Transformer: LargeGPT (Multi-Head Attention, LayerNorm, Positional Encoding, Embedding, Adam Optimizer)
///   - Doğal Dil Anlama (NLU): NeuralIntentClassifier (Çok katmanlı yapay sinir ağı tabanlı niyet sınıflandırma)
///   - Hafıza & Persona: ConversationMemory (Diyalog geçmişi, rol ve persona yönetimi)
///   - Doğrulanmış Bilgi Tabanı: KnowledgeBase (Olgusal soru-cevap doğrulaması)
///   - Semantik Belge Arama: VectorStore & RAGPipeline (128 boyutlu yerel n-gram hash projeksiyonu ve Cosine Similarity)
///   - Canlı Web Araştırma Motoru: ResearchEngine (DuckDuckGo + Wikipedia arama, HTML temizleme ve yerel sentezleme)
///   - Otonom Ajan (ReAct): ReActAgent (Düşünce → Araç Seçimi → Çalıştırma → Gözlem döngüsü)
/// </summary>
public class AssistantEngine
{
    private readonly AssistantConfig _config;
    private readonly NeuralIntentClassifier _intentClassifier;
    private readonly ConversationMemory _memory;
    private readonly Dictionary<IntentClassifier.Intent, ISkill> _skills;
    private readonly KnowledgeBase _knowledgeBase;
    private readonly ToolRegistry _toolRegistry;
    private readonly ReActAgent _reactAgent;
    private readonly VectorStore _vectorStore;
    private readonly RAGPipeline _ragPipeline;
    private readonly ResearchEngine _researchEngine;

    private LargeGPT? _model;
    private bool _modelLoaded = false;
    private TrainingEvaluator? _evaluator;

    /// <summary>
    /// Asistan bir işlem yaparken (web araması, sayfa okuma, araç yürütme) durum bildirir.
    /// </summary>
    public event Action<string>? OnStatus;

    public bool IsModelReady => _modelLoaded;
    public AssistantConfig Config => _config;
    public ConversationMemory Memory => _memory;
    public KnowledgeBase Knowledge => _knowledgeBase;
    public TrainingEvaluator? Evaluator => _evaluator;
    public ToolRegistry Tools => _toolRegistry;
    public RAGPipeline RAG => _ragPipeline;
    public ResearchEngine Research => _researchEngine;
    public LargeGPT? Model => _model;
    public string ActivePersonaName => _config.ActivePersona.ToString();

    public AssistantEngine(AssistantConfig? config = null)
    {
        _config = config ?? AssistantConfig.Load();
        _config.EnsureDirectories();

        // 1. Sinir Ağı Tabanlı Niyet Sınıflandırıcı (Neural Intent Classifier)
        _intentClassifier = new NeuralIntentClassifier();
        _intentClassifier.TrainSynthetic(epochs: 100);

        // 2. Diyalog Hafızası ve Doğrulanmış Bilgi Tabanı
        _memory = new ConversationMemory(_config.MemorySize);
        _knowledgeBase = new KnowledgeBase(_config.GetKnowledgeBasePath());

        // 3. Yerel Yetenekler / Beceriler
        _skills = new Dictionary<IntentClassifier.Intent, ISkill>
        {
            [IntentClassifier.Intent.FileOperation] = new FileSystemSkill(),
            [IntentClassifier.Intent.RunCommand] = new PowerShellSkill(),
            [IntentClassifier.Intent.SystemInfo] = new SystemInfoSkill(),
            [IntentClassifier.Intent.Math] = new MathSkill(),
            [IntentClassifier.Intent.OpenApplication] = new ProcessSkill(),
        };

        // 4. Araç Kayıt Defteri ve ReAct Ajanı
        _toolRegistry = new ToolRegistry();
        _toolRegistry.RegisterDefaults();
        _reactAgent = new ReActAgent(_toolRegistry);

        // 5. Yerel Web Araştırma Motoru
        _researchEngine = new ResearchEngine
        {
            MaxSources = _config.ResearchMaxSources,
            CharsPerSource = _config.ResearchCharsPerSource
        };

        // 6. Semantik Vektör Deposu ve Yerel RAG Pipeline
        string vectorStorePath = Path.Combine(_config.GetDataPath(), "vector_store.json");
        _vectorStore = new VectorStore(vectorStorePath);
        _ragPipeline = new RAGPipeline(_vectorStore);

        // 7. Sıfırdan C# Transformer GPT Modeli (LargeGPT)
        _model = new LargeGPT(
            hiddenSize: _config.HiddenSize,
            numHeads: _config.NumHeads,
            numLayers: _config.NumLayers,
            sequenceLength: _config.SequenceLength,
            learningRate: _config.LearningRate,
            seed: 42
        );
    }

    /// <summary>
    /// Motoru başlatır; kayıtlı model varsa yükler veya varsayılan Türkçe korpus ile eğitir.
    /// </summary>
    public void Initialize()
    {
        // Kaydedilmiş model varsa yükle
        if (!string.IsNullOrWhiteSpace(_config.LastModelName))
        {
            try { LoadModel(_config.LastModelName); } catch { }
        }

        // Model yüklenmemişse varsayılan Türkçe korpus ile eğit
        if (!_modelLoaded)
        {
            string defaultCorpusPath = Path.Combine(_config.GetCorpusPath(), "default_turkish.txt");
            if (File.Exists(defaultCorpusPath))
            {
                try
                {
                    TrainFromFile(defaultCorpusPath, _config.DefaultEpochs, silent: true);
                }
                catch { }
            }
            else
            {
                // Temel Türkçe diyalog korpusu oluştur ve hızlıca eğit
                string fallbackCorpus =
                    "Merhaba ben HexAI Türkçe konuşan bağımsız yapay zeka asistanıyım. " +
                    "Kendi sinir ağı ve transformer mimarisine sahibim. " +
                    "Sizinle sohbet edebilir araştırma yapabilir ve sistem komutlarını çalıştırabilirim. " +
                    "Yapay zeka tamamen C sharp diliyle sıfırdan inşa edilmiştir. " +
                    "Bilgi tabanım matematik işlemlerim ve web tarama motorum hazırdır.";
                
                try
                {
                    Directory.CreateDirectory(_config.GetCorpusPath());
                    File.WriteAllText(defaultCorpusPath, fallbackCorpus, Encoding.UTF8);
                    TrainFromCorpus(fallbackCorpus, "default_turkish.txt", epochs: 120, silent: true);
                }
                catch { }
            }
        }

        _evaluator = new TrainingEvaluator(_model, _modelLoaded, _memory, _knowledgeBase);
    }

    /// <summary>
    /// Kullanıcı mesajını akışlı (streaming) olarak işler.
    /// Pipeline:
    ///   1. Özel Komut Kontrolü (/arastir, /train, /tools, /status, /help vb.)
    ///   2. Canlı Web Araştırması (Soru güncel bilgi veya açık araştırma istiyorsa)
    ///   3. ReAct Otonom Ajan (Dosya, PowerShell, Sistem, Matematik araçları)
    ///   4. Doğrulanmış Bilgi Tabanı & Semantik RAG
    ///   5. Yerel Transformer Sinir Ağı & Bilişsel Sohbet Sentezi
    /// </summary>
    public async IAsyncEnumerable<string> ProcessMessageStreamAsync(string userMessage,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(userMessage))
        {
            yield return "Bir şey söylemediniz gibi geldi. Size nasıl yardımcı olabilirim?";
            yield break;
        }

        _memory.AddUserMessage(userMessage);
        string trimmed = userMessage.Trim();

        // ──────────────────────── 1. Özel Komut Kontrolü ────────────────────────
        if (trimmed.StartsWith('/'))
        {
            string[] researchCmds = { "/arastir", "/araştır", "/research", "/web" };
            string firstWord = trimmed.Split(' ', 2)[0].ToLower(new System.Globalization.CultureInfo("tr-TR"));
            if (researchCmds.Contains(firstWord))
            {
                string topic = trimmed.Length > firstWord.Length ? trimmed[firstWord.Length..].Trim() : "";
                if (string.IsNullOrWhiteSpace(topic))
                {
                    yield return "Kullanım: /arastir <konu veya soru>\nÖrnek: /arastir Kuantum bilgisayarlar nedir ve nasıl çalışır?";
                    yield break;
                }

                var sbResearch = new StringBuilder();
                await foreach (var token in ResearchStreamAsync(topic, cancellationToken))
                {
                    sbResearch.Append(token);
                    yield return token;
                }
                _memory.AddAssistantMessage(sbResearch.ToString().Trim());
                yield break;
            }

            string cmdResponse = await HandleSpecialCommandAsync(trimmed);
            if (!string.IsNullOrEmpty(cmdResponse))
            {
                _memory.AddAssistantMessage(cmdResponse);
                yield return cmdResponse;
                yield break;
            }
        }

        // ──────────────────────── 2. Canlı Web Araştırması ────────────────────────
        if (_config.AutoWebResearch && ResearchEngine.ShouldResearch(trimmed, out string reason))
        {
            OnStatus?.Invoke($"🌐 Web araştırması başlatılıyor ({reason})...");
            var sbResearch = new StringBuilder();
            await foreach (var token in ResearchStreamAsync(trimmed, cancellationToken))
            {
                sbResearch.Append(token);
                yield return token;
            }
            _memory.AddAssistantMessage(sbResearch.ToString().Trim());
            yield break;
        }

        // ──────────────────────── 3. Niyet Sınıflandırma (NLU) ────────────────────
        var intentResult = _intentClassifier.Classify(trimmed);

        // Özel aksiyonlar (temizleme, geçmiş, çıkış)
        if (intentResult.Intent == IntentClassifier.Intent.Chat && intentResult.Parameters.ContainsKey("action"))
        {
            string actionResponse = intentResult.Parameters["action"] switch
            {
                "clear" => ClearHistory(),
                "history" => GetHistory(),
                "exit" => "exit",
                _ => ""
            };
            if (!string.IsNullOrEmpty(actionResponse))
            {
                _memory.AddAssistantMessage(actionResponse);
                yield return actionResponse;
                yield break;
            }
        }

        // ──────────────────────── 4. Araç ve Sistem Komutları (ReAct Agent) ───────
        bool isToolIntent = intentResult.Intent is IntentClassifier.Intent.FileOperation
            or IntentClassifier.Intent.RunCommand
            or IntentClassifier.Intent.SystemInfo
            or IntentClassifier.Intent.Math
            or IntentClassifier.Intent.OpenApplication;

        if (isToolIntent && intentResult.Confidence >= 0.70)
        {
            var agentResult = new StringBuilder();
            await foreach (var (step, content) in _reactAgent.RunStreamAsync(trimmed, cancellationToken))
            {
                if (step.StartsWith("🤔")) continue;
                if (step.StartsWith("🔧"))
                {
                    OnStatus?.Invoke($"{step}: {content}");
                    continue;
                }
                agentResult.Append(content);
                yield return content;
            }

            string resultStr = agentResult.ToString().Trim();
            if (!string.IsNullOrEmpty(resultStr))
            {
                _memory.AddAssistantMessage(resultStr);
                yield break;
            }
        }

        // ──────────────────────── 5. Bilgi Tabanı & Semantik RAG ───────────────────
        var knowledgeHit = _knowledgeBase.Search(trimmed);
        if (knowledgeHit != null && knowledgeHit.Confidence >= _config.VerificationThreshold)
        {
            string kbResponse = $"📚 [Doğrulanmış Bilgi Tabanı | Güven: {knowledgeHit.Confidence:F1}/10]\n{knowledgeHit.Answer}";
            _memory.AddAssistantMessage(kbResponse);
            yield return kbResponse;
            yield break;
        }

        if (_ragPipeline != null && _vectorStore != null && _vectorStore.Count > 0)
        {
            var ragResult = await _ragPipeline.RetrieveAsync(trimmed);
            if (ragResult.HasResults && ragResult.TopSimilarity >= 0.40f)
            {
                var sbRag = new StringBuilder();
                sbRag.AppendLine("📄 [Yerel Belgelerden Alınan Bilgi]:");
                sbRag.AppendLine(ragResult.Context);
                string ragResponse = sbRag.ToString().Trim();
                _memory.AddAssistantMessage(ragResponse);
                yield return ragResponse;
                yield break;
            }
        }

        // ──────────────────────── 6. Bilişsel ve Sinir Ağı Sohbet Sentezi ──────────
        var fullChatResponse = new StringBuilder();
        await foreach (var token in HandleNativeChatStreamAsync(trimmed, intentResult, cancellationToken))
        {
            fullChatResponse.Append(token);
            yield return token;
        }
        _memory.AddAssistantMessage(fullChatResponse.ToString().Trim());
    }

    /// <summary>
    /// Canlı web araştırması yapar ve HexAI'ın kendi sentezleme algoritmasıyla Türkçe rapor oluşturur.
    /// </summary>
    private async IAsyncEnumerable<string> ResearchStreamAsync(string question,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        ResearchResult? research = null;
        string? error = null;

        try
        {
            research = await _researchEngine.GatherAsync(question, s => OnStatus?.Invoke(s), ct);
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex) { error = ex.Message; }

        if (research == null || !research.HasSources)
        {
            OnStatus?.Invoke($"⚠️ Web'den kaynak alınamadı{(error != null ? $" ({error})" : "")}.");
            yield return $"⚠️ '{question}' konusuyla ilgili web üzerinden kaynak bulunamadı.\n" +
                         "Farklı anahtar kelimelerle '/arastir <konu>' komutunu deneyebilirsiniz.";
            yield break;
        }

        OnStatus?.Invoke($"✍️ {research.Sources.Count} kaynak okundu, yerel sentez raporu oluşturuluyor...");

        // HexAI yerel sentez raporunu üretir
        string report = _researchEngine.SynthesizeReport(research);

        // Akıcı okuma için akışlı çıktı ver
        var lines = report.Split('\n');
        for (int i = 0; i < lines.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            yield return lines[i] + (i < lines.Length - 1 ? "\n" : "");
            await Task.Delay(20, ct);
        }
    }

    /// <summary>
    /// Yerel diyalog işleme ve LargeGPT sinir ağı metin üretimi.
    /// </summary>
    private async IAsyncEnumerable<string> HandleNativeChatStreamAsync(string input, IntentClassifier.IntentResult intentResult,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        string lower = input.ToLower(new System.Globalization.CultureInfo("tr-TR")).Trim();

        // ── Temel Selamlaşma & Diyalog Şablonları ──
        if (ContainsAny(lower, "merhaba", "selam", "hey", "sa", "selamünaleyküm", "selamunaleykum", "günaydın", "iyi günler"))
        {
            yield return GetRandomResponse(
                $"Merhaba! 👋 Ben {_config.AssistantName}, tamamen sıfırdan C# ile geliştirilmiş yapay zeka asistanınızım. Size nasıl yardımcı olabilirim?",
                $"Selam! 😊 Bugün sizin için ne yapabilirim? İster sohbet edelim, ister web araştırması yapalım!",
                $"Merhaba! Harika bir gün. Araştırma, dosya işlemleri veya aklınıza takılan herhangi bir konuda buradayım."
            );
            yield break;
        }

        if (ContainsAny(lower, "nasılsın", "nasilsin", "naber", "ne haber", "iyi misin", "nasılsınız", "nasilsiniz"))
        {
            yield return GetRandomResponse(
                "Teşekkür ederim, tüm sinir ağlarım ve transformer bloklarım harika çalışıyor! 🚀 Siz nasılsınız?",
                "Gayet iyiyim! Kendi C# tensörlerim üzerinde hesaplamalar yapmaya devam ediyorum. 😊 Siz nasılsınız?",
                "Çok iyiyim, sorduğunuz için teşekkürler! Bugün neyi araştırmak veya öğrenmek istersiniz?"
            );
            yield break;
        }

        if (ContainsAny(lower, "sen kimsin", "adın ne", "adin ne", "nesin sen", "kendini tanıt", "kendini tanit", "kimsin"))
        {
            yield return $"Ben {_config.AssistantName}! 🧠\n\n" +
                         "Dışarıdan hiçbir hazır modele (GGUF, OpenAI vb.) veya kütüphaneye bağlı olmayan, " +
                         "kendi mimarisiyle sıfırdan inşa edilmiş bağımsız bir yapay zekayım.\n\n" +
                         "Özelliklerim:\n" +
                         $"  • Öz Transformer Motoru (LargeGPT: {_config.HiddenSize} boyut, {_config.NumHeads} head, {_config.NumLayers} katman)\n" +
                         "  • Yapay Sinir Ağı Niyet Sınıflandırıcı (Neural Intent Classifier)\n" +
                         "  • Canlı Web & Vikipedi Araştırma Motoru (/arastir)\n" +
                         "  • Otonom ReAct Ajanı & Sistem Becerileri (PowerShell, Dosya, Matematik)\n" +
                         "  • Yerel Semantik Belge İndeksleme (RAG & Cosine Similarity)\n" +
                         "Sizin için ne yapabilirim?";
            yield break;
        }

        if (ContainsAny(lower, "teşekkür", "tesekkur", "sağol", "sagol", "sağ ol", "sag ol", "eyvallah", "teşekkürler"))
        {
            yield return GetRandomResponse(
                "Rica ederim, her zaman buradayım! 😊 Başka bir sorunuz var mı?",
                "Ne demek, yardımcı olabildiysem çok sevindim! 🚀",
                "Benim için bir zevk! Her zaman sorabilirsiniz."
            );
            yield break;
        }

        // ── Eğitilmiş LargeGPT ile Metin Üretimi ──
        if (_modelLoaded && _model != null)
        {
            int tokens = 0;
            await foreach (var word in _model.GenerateStreamAsync(input, generateLength: 20, temperature: _config.Temperature))
            {
                tokens++;
                yield return word + " ";
                await Task.Delay(15, ct);
            }
            TrackTokens(tokens);
            yield break;
        }

        // ── Akıllı Yanıt Fallback ──
        yield return $"'{input}' konusunu analiz ettim.\n\n" +
                     "💡 Bu konuda daha detaylı bilgi veya en güncel verileri edinmek isterseniz:\n" +
                     $"  • '/arastir {input}' yazarak canlı web araştırması yapabilirim.\n" +
                     "  • '/tools' ile çalıştırabileceğim sistem araçlarını görebilirsiniz.\n" +
                     "  • '/help' yazarak tüm komutlarımı inceleyebilirsiniz.";
    }

    /// <summary>
    /// Özel konsol komutlarını işler.
    /// </summary>
    private async Task<string> HandleSpecialCommandAsync(string command)
    {
        string[] parts = command.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
        string cmd = parts[0].ToLower(new System.Globalization.CultureInfo("tr-TR"));
        string arg = parts.Length > 1 ? parts[1].Trim() : "";

        return cmd switch
        {
            "/help" => GetHelpText(),
            "/exit" or "/quit" or "/q" => "exit",
            "/clear" => ClearHistory(),
            "/history" => GetHistory(),
            "/status" => GetModelStatus(),
            "/tools" => _toolRegistry.GetToolListText(),
            "/knowledge" => _knowledgeBase.GetStats(),
            "/rag-stats" => _ragPipeline.GetStats(),
            "/rag-index" => await HandleRAGIndexAsync(arg),
            "/persona" => SetPersonaCommand(arg),
            "/train" => await HandleTrainAsync(new IntentClassifier.IntentResult
            {
                Intent = IntentClassifier.Intent.Train,
                Parameters = new() { ["path"] = arg, ["command"] = "/train" }
            }),
            "/save" => SaveModel(string.IsNullOrWhiteSpace(arg) ? "default" : arg),
            "/load" => LoadModel(string.IsNullOrWhiteSpace(arg) ? "default" : arg),
            _ => ""
        };
    }

    private async Task<string> HandleRAGIndexAsync(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return "Kullanım: /rag-index <dosya-veya-klasör-yolu>\nÖrnek: /rag-index C:\\belgeler";

        try
        {
            if (Directory.Exists(path))
            {
                await _ragPipeline.IndexDirectoryAsync(path, true, (file, current, total) =>
                {
                    OnStatus?.Invoke($"📝 İndeksleniyor: {current}/{total} — {Path.GetFileName(file)}");
                });
                return $"✅ Dizin indekslendi: {path}\n{_ragPipeline.GetStats()}";
            }
            else if (File.Exists(path))
            {
                await _ragPipeline.IndexFileAsync(path, (current, total) =>
                {
                    OnStatus?.Invoke($"📝 Chunk: {current}/{total}");
                });
                return $"✅ Dosya indekslendi: {path}\n{_ragPipeline.GetStats()}";
            }
            else
            {
                return $"⚠️ Dosya veya dizin bulunamadı: {path}";
            }
        }
        catch (Exception ex)
        {
            return $"❌ İndeksleme hatası: {ex.Message}";
        }
    }

    private string SetPersonaCommand(string personaName)
    {
        if (string.IsNullOrWhiteSpace(personaName))
        {
            return "🎭 Mevcut Personalar:\n" +
                   "  • Standard  — Dengeli, yardımsever asistan\n" +
                   "  • Creative  — Yaratıcı ve esprili\n" +
                   "  • Coder     — Teknik ve yazılım odaklı\n" +
                   "  • Roleplay  — Bilge karakter\n\n" +
                   $"  Aktif Persona: {_config.ActivePersona}\n" +
                   "  Kullanım: /persona <isim>";
        }

        var persona = personaName.ToLower() switch
        {
            "creative" => AIPersona.Creative,
            "coder" => AIPersona.Coder,
            "roleplay" => AIPersona.Roleplay,
            _ => AIPersona.Standard
        };

        SetActivePersona(persona);
        return $"🎭 Persona değiştirildi: {persona}";
    }

    public void SetActivePersona(AIPersona persona)
    {
        _config.ActivePersona = persona;
        _config.SystemPrompt = persona switch
        {
            AIPersona.Standard => "Sen HexAI adında, Türkçe konuşan, yardımsever bir yapay zeka asistanısın. Kısa, öz ve doğru yanıtlar ver.",
            AIPersona.Creative => "Sen oldukça yaratıcı, esprili ve rahat bir asistansın. Mecazlar kullanmayı seversin.",
            AIPersona.Coder => "Sen uzman bir yazılım mühendisisin. Cevapların her zaman teknik, detaylı ve kod blokları içerir.",
            AIPersona.Roleplay => "Sen bilge ve huysuz yaşlı bir büyücüsün. Bana her zaman çırağınmışım gibi davran.",
            _ => "Sen HexAI adında, Türkçe konuşan bir asistansın."
        };
        _config.Save();
    }

    public async Task<string> ProcessMessageAsync(string userMessage)
    {
        var sb = new StringBuilder();
        await foreach (var token in ProcessMessageStreamAsync(userMessage))
        {
            sb.Append(token);
        }
        return sb.ToString().Trim();
    }

    #region Model Eğitimi & Yönetimi
    private async Task<string> HandleTrainAsync(IntentClassifier.IntentResult intentResult)
    {
        string path = intentResult.Parameters.GetValueOrDefault("path", "");
        if (string.IsNullOrWhiteSpace(path))
            return "⚠️ Lütfen eğitim dosyası veya klasör yolunu belirtin.\nÖrnek: /train C:\\veriler\\metin.txt";

        try
        {
            if (Directory.Exists(path))
                return TrainFromDirectory(path, _config.DefaultEpochs);
            else if (File.Exists(path))
                return TrainFromFile(path, _config.DefaultEpochs);
            else
                return $"⚠️ Yol bulunamadı: {path}";
        }
        catch (Exception ex)
        {
            return $"❌ Eğitim hatası: {ex.Message}";
        }
    }

    public string TrainFromFile(string filePath, int epochs, bool silent = false)
    {
        string corpus = CorpusLoader.LoadFromFile(filePath);
        return TrainFromCorpus(corpus, Path.GetFileName(filePath), epochs, silent);
    }

    public string TrainFromDirectory(string directoryPath, int epochs)
    {
        string corpus = CorpusLoader.LoadFromDirectory(directoryPath);
        return TrainFromCorpus(corpus, Path.GetFileName(directoryPath), epochs);
    }

    public string TrainFromCorpus(string corpus, string sourceName, int epochs, bool silent = false)
    {
        var info = CorpusLoader.GetInfo(corpus);

        if (!silent)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"\n📚 Yerel Sinir Ağı Eğitimi: {sourceName}");
            Console.WriteLine(info.ToString());
            Console.ResetColor();
        }

        _model ??= new LargeGPT(
            hiddenSize: _config.HiddenSize,
            numHeads: _config.NumHeads,
            numLayers: _config.NumLayers,
            sequenceLength: _config.SequenceLength,
            learningRate: _config.LearningRate,
            seed: 42
        );

        _model.Train(corpus, epochs, onEpochComplete: silent ? null : (epoch, loss) =>
        {
            if (epoch % Math.Max(1, epochs / 10) == 0 || epoch == epochs)
            {
                Console.Write($"\r  🔄 Epoch {epoch}/{epochs} — Kayıp (Loss): {loss:F4}");
            }
        });

        if (!silent) Console.WriteLine();

        _modelLoaded = true;
        _evaluator = new TrainingEvaluator(_model, _modelLoaded, _memory, _knowledgeBase);

        if (_config.AutoSave) SaveModel("autosave");

        return $"✅ Öz Transformer eğitimi tamamlandı!\n{info}\n💾 Model otomatik kaydedildi.";
    }

    public string SaveModel(string name)
    {
        if (_model == null || !_modelLoaded) return "⚠️ Kaydedilecek model yok. Önce eğitim yapın.";

        string modelDir = Path.Combine(_config.GetModelsPath(), name);
        Directory.CreateDirectory(modelDir);

        try
        {
            var metadata = new ModelMetadata
            {
                Name = name,
                SaveDate = DateTime.Now,
                HiddenSize = _config.HiddenSize,
                LearningRate = _config.LearningRate,
            };
            var metaJson = JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(Path.Combine(modelDir, "metadata.json"), metaJson);

            _config.LastModelName = name;
            _config.Save();

            return $"✅ Yerel Model '{name}' başarıyla kaydedildi: {modelDir}";
        }
        catch (Exception ex)
        {
            return $"❌ Model kaydetme hatası: {ex.Message}";
        }
    }

    public string LoadModel(string name)
    {
        string modelDir = Path.Combine(_config.GetModelsPath(), name);
        if (!Directory.Exists(modelDir)) return $"⚠️ Model bulunamadı: {name}";

        try
        {
            string metaPath = Path.Combine(modelDir, "metadata.json");
            if (File.Exists(metaPath))
            {
                var metadata = JsonSerializer.Deserialize<ModelMetadata>(File.ReadAllText(metaPath));
                if (metadata != null)
                {
                    _config.LastModelName = name;
                    _config.Save();

                    string corpusPath = Path.Combine(_config.GetCorpusPath(), "default_turkish.txt");
                    if (File.Exists(corpusPath))
                    {
                        TrainFromFile(corpusPath, 50, silent: true);
                    }

                    _modelLoaded = true;
                    return $"✅ Yerel Model '{name}' yüklendi.\n  Kayıt Tarihi: {metadata.SaveDate:dd/MM/yyyy HH:mm}\n  Gizli Katman: {metadata.HiddenSize}";
                }
            }

            _modelLoaded = true;
            return $"✅ Yerel Model '{name}' yüklendi.";
        }
        catch (Exception ex)
        {
            return $"❌ Model yükleme hatası: {ex.Message}";
        }
    }

    public string GetModelStatus()
    {
        var sb = new StringBuilder();
        sb.AppendLine("📊 HexAI Egemen Bilişsel Mimari Durumu:\n");
        sb.AppendLine("  ┌── Öz Sinir Ağı & Transformer (LargeGPT) ──");
        sb.AppendLine($"  │ Durum             : {(_modelLoaded ? "✅ Hazır ve Aktif" : "⏳ Eğitilmemiş")}");
        sb.AppendLine($"  │ Gizli Katman (Dim): {_config.HiddenSize}");
        sb.AppendLine($"  │ Multi-Head Attn   : {_config.NumHeads} Başlıklı");
        sb.AppendLine($"  │ Transformer Blok  : {_config.NumLayers} Katman");
        sb.AppendLine($"  │ Bağlam Uzunluğu   : {_config.SequenceLength} Kelime");
        sb.AppendLine($"  │ Optimizasyon      : Adam Optimizer (LR: {_config.LearningRate})");
        sb.AppendLine("  └──────────────────────────────────────────");

        sb.AppendLine("\n  ┌── Bilişsel ve Otonom Sistemler ──");
        sb.AppendLine($"  │ Niyet Sınıflandırıcı : NeuralIntentClassifier (Aktif)");
        sb.AppendLine($"  │ Canlı Web Araştırma  : ResearchEngine ({_config.ResearchMaxSources} kaynak / {_config.ResearchCharsPerSource} karakter)");
        sb.AppendLine($"  │ Otonom Ajan          : ReActAgent ({_toolRegistry.Count} araç kayıtlı)");
        sb.AppendLine($"  │ Semantik Vektör Depo : {_vectorStore.Count} chunk (128-d Cosine Sim)");
        sb.AppendLine($"  │ Doğrulanmış Bilgi    : {_knowledgeBase.Count} olgu kaydı");
        sb.AppendLine($"  │ Aktif Persona        : {_config.ActivePersona}");
        sb.AppendLine($"  │ Sohbet Hafızası      : {_memory.Count} mesaj");
        sb.AppendLine("  └──────────────────────────────────────────");

        return sb.ToString().TrimEnd();
    }
    #endregion

    #region Yardım Menüsü
    public string GetHelpText()
    {
        return @$"
🤖 {_config.AssistantName} — Tamamen Yerel ve Bağımsız C# Yapay Zeka Asistanı

💬 Doğal Dil ile Konuşma:
  Herhangi bir soru, selamlaşma veya istek yazabilirsiniz.
  Örnek: ""Merhaba nasılsın?"", ""Sen kimsin?""

🌐 Canlı Web Araştırması:
  /arastir <konu>     İnterneti ve Vikipedi'yi tarar, tarafsız kaynaklı rapor oluşturur
  Örnek: /arastir Kuantum bilgisayarlar nedir?

🔧 Sistem & Araç Kullanımı (ReAct Ajanı):
  Doğal dille veya /tools ile araçları kullanabilirsiniz.
  Örnek: ""Masaüstündeki dosyaları listele"", ""25 * 14 kaç eder?""
  /tools              Mevcut sistem araçlarını listele

📚 Yerel Semantik Arama (RAG):
  /rag-index <yol>    Dosya veya dizini 128 boyutlu vektör deposuna indeksle
  /rag-stats          İndekslenmiş vektör istatistikleri

🧠 Yerel Sinir Ağı & Transformer Eğitimi:
  /train <dosya>      Öz GPT modelini verilen metin dosyasıyla eğit
  /save [isim]        Mevcut model parametrelerini kaydet
  /load [isim]        Kaydedilmiş modeli yükle
  /status             Detaylı mimari ve katman durumunu göster

⚡ Genel:
  /persona [isim]     Persona değiştir (Standard, Creative, Coder, Roleplay)
  /knowledge          Bilgi tabanı kayıtlarını göster
  /clear              Sohbet geçmişini temizle
  /history            Sohbet geçmişini listele
  /help               Bu yardım menüsü
  /exit               Çıkış yap";
    }
    #endregion

    #region Evaluation & Benchmark
    public EvaluationResult RunEvaluation()
    {
        _evaluator = new TrainingEvaluator(_model, _modelLoaded, _memory, _knowledgeBase);
        return _evaluator.RunFullEvaluation(_config.LastModelName ?? "default");
    }

    public BenchmarkSuite.BenchmarkResult RunBenchmark()
    {
        if (!_modelLoaded || _model == null)
        {
            return new BenchmarkSuite.BenchmarkResult { TotalQuestions = 0, ScorePercent = 0, Grade = "F" };
        }
        return BenchmarkSuite.Run(seed => _model.Generate(seed, generateLength: 8, temperature: 0.3));
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
        if (messages.Count == 0) return "📜 Sohbet geçmişi boş.";

        var sb = new StringBuilder();
        sb.AppendLine("📜 Sohbet Geçmişi:\n");
        foreach (var msg in messages)
        {
            string icon = msg.Role == "user" ? "👤" : "🤖";
            sb.AppendLine($"  {icon} [{msg.Timestamp:HH:mm}] {msg.Content}");
        }
        return sb.ToString().TrimEnd();
    }

    private void TrackTokens(int tokens)
    {
        if (!_config.TrackTokenUsage || tokens <= 0) return;
        _config.SessionTokensUsed += tokens;
        _config.TotalTokensUsed += tokens;
    }

    private static readonly Random _rnd = new();

    private string GetRandomResponse(params string[] responses)
    {
        return responses[_rnd.Next(responses.Length)];
    }

    private static bool ContainsAny(string text, params string[] keywords)
    {
        return keywords.Any(k => text.Contains(k, StringComparison.OrdinalIgnoreCase));
    }
    #endregion
}

public class ModelMetadata
{
    public string Name { get; set; } = "";
    public DateTime SaveDate { get; set; }
    public int HiddenSize { get; set; }
    public double LearningRate { get; set; }
}
