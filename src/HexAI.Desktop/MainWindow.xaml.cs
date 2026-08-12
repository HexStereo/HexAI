using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Controls;
using HexAI.Assistant;
using HexAI.Desktop.ViewModels;
using HexAI.Desktop.Services;

namespace HexAI.Desktop;

public partial class MainWindow : Window
{
    private AssistantEngine _engine = null!;
    private VoiceManager _voiceManager = null!;
    private ChatSessionManager _sessionManager = null!;
    
    public ObservableCollection<ChatSession> Sessions { get; set; } = new();

    private ChatSession _activeSession = null!;
    private bool _isVoiceListening = false;

    public MainWindow()
    {
        InitializeComponent();
        SessionList.ItemsSource = Sessions;
        _sessionManager = new ChatSessionManager();
        Loaded += MainWindow_Loaded;
    }

    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        LoadPastSessions();
        _voiceManager = new VoiceManager();
        if (_voiceManager.IsVoiceSupported)
        {
            VoiceButton.IsEnabled = true;
        }

        // Motoru arka planda başlat
        await Task.Run(() =>
        {
            _engine = new AssistantEngine();
            _engine.Initialize();
        });

        LoadingIndicator.Visibility = Visibility.Collapsed;
        InputBox.IsEnabled = true;
        SendButton.IsEnabled = true;

        if (_activeSession.Messages.Count == 0)
        {
            AddMessage("Merhaba! 👋 Ben HexAI, senin kişisel masaüstü asistanın. Sana nasıl yardımcı olabilirim?", isUser: false, session: _activeSession);
        }
        
        InputBox.Focus();
    }

    private async void SendButton_Click(object sender, RoutedEventArgs e)
    {
        await ProcessInput();
    }

    private async void InputBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
        {
            await ProcessInput();
        }
    }

    private async Task ProcessInput()
    {
        string input = InputBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(input)) return;

        InputBox.Clear();
        InputBox.IsEnabled = false;
        SendButton.IsEnabled = false;

        // O anki aktif session'ı yakala (kullanıcı AI düşünürken başka sohbete geçerse diye)
        ChatSession currentSession = _activeSession;

        // Update title if new session
        if (currentSession.Title == "Yeni Sohbet" || currentSession.Title == "")
        {
            currentSession.Title = input.Length > 25 ? input.Substring(0, 25) + "..." : input;
            _sessionManager.SaveSession(currentSession);
        }

        AddMessage(input, isUser: true, session: currentSession);

        // UI donmasın diye motoru task içinde çağırıyoruz
        string response = await Task.Run(async () => await _engine.ProcessMessageAsync(input));

        if (response == "exit")
        {
            Application.Current.Shutdown();
            return;
        }

        await TypewriterEffect(response, currentSession);

        if (_engine.Config.VoiceOutputEnabled)
        {
            _voiceManager.SpeakAsync(response);
        }

        InputBox.IsEnabled = true;
        SendButton.IsEnabled = true;
        InputBox.Focus();
    }

    private async void VoiceButton_Click(object sender, RoutedEventArgs e)
    {
        if (!_voiceManager.IsVoiceSupported) return;

        if (_isVoiceListening)
        {
            _voiceManager.StopListening();
            _isVoiceListening = false;
            VoiceButton.Foreground = Brushes.White;
            VoiceButton.Background = new SolidColorBrush(Color.FromRgb(29, 22, 58)); // #1d163a
            return;
        }

        _isVoiceListening = true;
        VoiceButton.Foreground = Brushes.White;
        VoiceButton.Background = new SolidColorBrush(Color.FromRgb(220, 20, 60)); // Crimson Red for recording
        InputBox.Text = "Dinleniyor...";

        string result = await _voiceManager.ListenAsync();
        
        _isVoiceListening = false;
        VoiceButton.Foreground = Brushes.White;
        VoiceButton.Background = new SolidColorBrush(Color.FromRgb(29, 22, 58));
        
        if (!string.IsNullOrWhiteSpace(result))
        {
            InputBox.Text = result;
            await ProcessInput();
        }
        else
        {
            InputBox.Clear();
        }
    }

    private void AddMessage(string text, bool isUser, ChatSession session)
    {
        var msg = new ChatMessage { Text = text, IsUser = isUser };
        session.Messages.Add(msg);
        _sessionManager.SaveSession(session);
        
        if (session == _activeSession)
        {
            ScrollToBottom();
        }
    }

    private async Task TypewriterEffect(string text, ChatSession session)
    {
        var msg = new ChatMessage { Text = "", IsUser = false };
        session.Messages.Add(msg);
        
        if (session == _activeSession) ScrollToBottom();

        int delay = _engine.Config.TypewriterDelayMs;

        foreach (char c in text)
        {
            msg.Text += c;
            
            if (session == _activeSession) ScrollToBottom();
            
            if (delay > 0)
            {
                await Task.Delay(c switch
                {
                    '.' or '!' or '?' => delay * 3,
                    ',' => delay * 2,
                    _ => delay
                });
            }
        }
        
        _sessionManager.SaveSession(session);
    }

    private void ScrollToBottom()
    {
        ChatScrollViewer.ScrollToEnd();
    }

    protected override void OnClosed(EventArgs e)
    {
        _voiceManager?.Dispose();
        base.OnClosed(e);
    }

    // --- Backend Integration (Sidebar) ---
    private void LoadPastSessions()
    {
        var saved = _sessionManager.LoadAllSessions();
        foreach (var s in saved) Sessions.Add(s);
        
        if (Sessions.Count > 0)
        {
            SetActiveSession(Sessions[0]);
        }
        else
        {
            CreateNewSession();
        }
    }

    private void CreateNewSession()
    {
        var newSession = new ChatSession();
        Sessions.Insert(0, newSession);
        SetActiveSession(newSession);
        SessionList.SelectedItem = newSession;
    }

    private void SetActiveSession(ChatSession session)
    {
        _activeSession = session;
        
        // Sadece aktif seansın mesajlarını göster (Global liste yok)
        ChatList.ItemsSource = _activeSession.Messages;
        
        if (_engine != null)
        {
            _engine.Memory.Clear();
            // Engine'in hafızasına geçmişi yükle
            foreach (var m in session.Messages)
            {
                if (m.IsUser) _engine.Memory.AddUserMessage(m.Text);
                else _engine.Memory.AddAssistantMessage(m.Text);
            }
        }
        
        ScrollToBottom();
    }

    private void NewChatButton_Click(object sender, RoutedEventArgs e)
    {
        CreateNewSession();
        AddMessage("Merhaba! 👋 Ben HexAI, yeni bir sohbete başladık. Ne sormak istersin?", isUser: false, session: _activeSession);
    }

    private void SessionList_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (SessionList.SelectedItem is ChatSession selected)
        {
            if (selected != _activeSession)
            {
                SetActiveSession(selected);
            }
        }
    }

    private void DeleteSession_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button button && button.DataContext is ChatSession session)
        {
            // Diskten sil
            _sessionManager.DeleteSession(session.Id);
            // Listeden sil
            Sessions.Remove(session);
            
            // Eğer sildiğimiz sohbet aktif sohbet ise başka birine geç
            if (_activeSession == session)
            {
                if (Sessions.Count > 0)
                {
                    SetActiveSession(Sessions[0]);
                }
                else
                {
                    CreateNewSession();
                }
            }
        }
    }

    private void SettingsButton_Click(object sender, RoutedEventArgs e)
    {
        if (_engine != null)
        {
            string status = _engine.GetModelStatus();
            MessageBox.Show(status, "HexAI Model Durumu", MessageBoxButton.OK, MessageBoxImage.Information);
        }
    }

    // --- Custom Title Bar Controls ---
    private void TitleBar_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (e.ClickCount == 2)
        {
            if (WindowState == WindowState.Normal) WindowState = WindowState.Maximized;
            else WindowState = WindowState.Normal;
        }
        else
        {
            DragMove();
        }
    }

    private void MinimizeButton_Click(object sender, RoutedEventArgs e)
    {
        WindowState = WindowState.Minimized;
    }

    private void CloseButton_Click(object sender, RoutedEventArgs e)
    {
        Application.Current.Shutdown();
    }
}
