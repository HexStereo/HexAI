using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using HexAI.Desktop.ViewModels;

namespace HexAI.Desktop.Services;

public class ChatSessionManager
{
    private readonly string _sessionsDir;

    public ChatSessionManager()
    {
        string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        _sessionsDir = Path.Combine(userProfile, ".hexai", "sessions");
        
        if (!Directory.Exists(_sessionsDir))
        {
            Directory.CreateDirectory(_sessionsDir);
        }
    }

    public List<ChatSession> LoadAllSessions()
    {
        var sessions = new List<ChatSession>();
        var files = Directory.GetFiles(_sessionsDir, "*.json");

        foreach (var file in files)
        {
            try
            {
                var json = File.ReadAllText(file);
                var session = JsonSerializer.Deserialize<ChatSession>(json);
                if (session != null)
                {
                    sessions.Add(session);
                }
            }
            catch
            {
                // Ignore corrupted sessions
            }
        }

        // En yeniler en üstte
        return sessions.OrderByDescending(s => s.CreatedAt).ToList();
    }

    public void SaveSession(ChatSession session)
    {
        if (session == null || string.IsNullOrWhiteSpace(session.Id)) return;

        string filePath = Path.Combine(_sessionsDir, $"{session.Id}.json");
        try
        {
            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(session, options);
            File.WriteAllText(filePath, json);
        }
        catch
        {
            // Fail silently
        }
    }

    public void DeleteSession(string sessionId)
    {
        if (string.IsNullOrWhiteSpace(sessionId)) return;

        string filePath = Path.Combine(_sessionsDir, $"{sessionId}.json");
        if (File.Exists(filePath))
        {
            try
            {
                File.Delete(filePath);
            }
            catch { }
        }
    }
}
