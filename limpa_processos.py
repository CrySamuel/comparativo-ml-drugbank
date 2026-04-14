import psutil
import time

# Lista de processos inúteis que identificamos nos seus prints
PROCESSOS_ALVO = [
    "brave.exe",
    "whatsapp.exe",
    "linkedin.exe",
    "discord.exe",
    "spotify.exe",
    "chrome.exe",
    "steam.exe",
    "steamwebhelper.exe",
    "epicgameslauncher.exe",
    "epicwebhelper.exe",
    "riotclientux.exe",
    "riotclientservices.exe",
    "riotclientcrashhandler.exe",
    "onedrive.exe",
    "steamclientbootstrapper.exe"
]

# Trava de segurança: Processos que o script é PROIBIDO de fechar
PROCESSOS_PROTEGIDOS = [
    "powershell.exe",  # Seu bot FinAI
    "pwsh.exe",        # PowerShell Core (garantia extra)
    "code.exe",        # VS Code (Seu TCC)
    "python.exe",      # O motor do seu TCC e do script
    "cmd.exe"
]

def liberar_recursos_seguro():
    print("Iniciando a faxina de processos...\n")
    processos_encerrados = 0

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            nome_processo = proc.info['name'].lower() if proc.info['name'] else ""
            
            # Se por acaso o nome do processo bater com a lista de protegidos, pula na hora
            if nome_processo in PROCESSOS_PROTEGIDOS:
                continue

            if nome_processo in PROCESSOS_ALVO:
                proc.terminate()
                proc.wait(timeout=3)
                print(f"[FECHADO] {nome_processo} (PID: {proc.info['pid']})")
                processos_encerrados += 1
                
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            pass
        except psutil.TimeoutExpired:
            proc.kill()
            print(f"[FORÇADO] {nome_processo} (PID: {proc.info['pid']})")
            processos_encerrados += 1
        except Exception:
            pass

    print(f"\nResumo: {processos_encerrados} processos de jogos e navegadores foram encerrados.")
    print("-> PowerShell (FinAI) intacto.")
    print("-> VS Code (TCC) intacto.")
    print("A máquina está livre para as 5 horas de treino. Manda ver!")

if __name__ == "__main__":
    liberar_recursos_seguro()