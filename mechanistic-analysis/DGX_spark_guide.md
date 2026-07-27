# Guia de Acesso — APIs de LLM do Cluster DGX Spark

> **Para quem é este guia:** qualquer pessoa do grupo que queira usar os modelos de linguagem
> hospedados no nosso cluster de 2× DGX Spark, a partir do seu próprio computador.
> Não é preciso conta nas máquinas nem conhecimento de infraestrutura.
>
> **Última atualização:** 2026-07-18

---

## 1. O que está disponível

O cluster serve **dois modelos simultaneamente**, atrás de uma única API compatível com a da
OpenAI — qualquer biblioteca/ferramenta que fale esse protocolo funciona (SDK da OpenAI,
LangChain, LiteLLM, Continue, Cursor, etc.). Você escolhe o modelo pelo campo `model` da requisição.

| Nome do modelo (campo `model`) | O que é | Contexto | Vocação |
|---|---|---|---|
| `nemotron-3-super` | NVIDIA Nemotron 3 Super 120B (MoE, *reasoning*) | **256K tokens** | Raciocínio complexo, matemática, código, **documentos longos** |
| `gemma-4-31b` | Google Gemma 4 31B (denso, **multimodal**) | 32K tokens | Respostas rápidas, tarefas gerais, **aceita imagens** |

Particularidades:

- O **Nemotron pensa antes de responder** — a resposta começa com o raciocínio dele.
  Use `max_tokens` generoso (≥ 1024) para não cortar a resposta no meio do raciocínio.
- O **Gemma aceita imagens** no formato padrão da OpenAI (content com `image_url`).
- Velocidades típicas: Nemotron ~16 tok/s · Gemma ~13 tok/s. A **primeira requisição após
  uma recarga do modelo pode demorar ~30 s** (compilação); as seguintes são normais.

## 2. O que você precisa (peça ao operador)

1. **A chave da API** — uma string longa; entra no cabeçalho `Authorization` de toda requisição.
2. **Acesso SSH ao spark1** (só se você for acessar de fora da rede da USP — ver seção 3):
   o operador precisa autorizar sua chave SSH pública no servidor. Envie a ele o conteúdo do
   seu `~/.ssh/id_ed25519.pub` (crie com `ssh-keygen -t ed25519` se não tiver).

> ⚠️ **Não compartilhe a chave da API fora do grupo** e não a coloque em código versionado.
> Use variável de ambiente: `export DGX_API_KEY="<chave>"`.

## 3. Conectando — escolha seu cenário

### Cenário A — você está DENTRO da rede da USP (cabo/wifi institucional ou VPN da USP)

Use a API direto, sem túnel:

```
Base URL:  http://200.144.205.82:8000/v1
```

Teste rápido:
```bash
curl http://200.144.205.82:8000/health
# esperado: {"status": "ok"}
```
Se o comando acima der timeout mesmo dentro da USP, siga para o Cenário B (algumas sub-redes
institucionais também filtram a porta 8000).

### Cenário B — você está FORA da USP (casa, escritório, 4G)

O firewall da USP **bloqueia a porta 8000 vinda de fora**, mas a porta 1232 (SSH) passa.
A solução é um túnel SSH que traz a API para o `localhost` da sua máquina.

**Passo único de configuração** (depois de o operador autorizar sua chave SSH):

```bash
# deixe rodando num terminal (reconecta sozinho se a conexão cair):
while true; do
  ssh -p 1232 -N -o ServerAliveInterval=15 -o ConnectTimeout=10 \
      -L 8000:localhost:8000 nvidia@200.144.205.82
  sleep 5
done
```

Ou, mais elegante, com autossh (macOS: `brew install autossh` / Ubuntu: `apt install autossh`):
```bash
autossh -M 0 -f -p 1232 -N -o ServerAliveInterval=15 -L 8000:localhost:8000 nvidia@200.144.205.82
```

Com o túnel de pé, sua Base URL passa a ser:
```
Base URL:  http://localhost:8000/v1
```

Teste: `curl http://localhost:8000/health` → `{"status": "ok"}`

## 4. Usando a API

Nos exemplos abaixo, use a Base URL do seu cenário (A ou B). Todos assumem
`export DGX_API_KEY="<sua-chave>"`.

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DGX_API_KEY" \
  -d '{
    "model": "gemma-4-31b",
    "messages": [{"role": "user", "content": "Explique RDMA em uma frase."}],
    "max_tokens": 200
  }'
```

### Python (SDK da OpenAI)

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="http://localhost:8000/v1",   # ou http://200.144.205.82:8000/v1 dentro da USP
    api_key=os.environ["DGX_API_KEY"],
)

resp = client.chat.completions.create(
    model="nemotron-3-super",              # ou "gemma-4-31b"
    messages=[{"role": "user", "content": "Prove que raiz de 2 é irracional."}],
    max_tokens=2048,
)
print(resp.choices[0].message.content)
```

### Python com streaming (tokens chegando ao vivo)

```python
stream = client.chat.completions.create(
    model="nemotron-3-super",
    messages=[{"role": "user", "content": "Escreva um resumo sobre atenção em transformers."}],
    max_tokens=1024,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### Imagem no Gemma (multimodal)

```python
import base64
b64 = base64.b64encode(open("figura.png", "rb").read()).decode()

resp = client.chat.completions.create(
    model="gemma-4-31b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Descreva esta figura."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
    max_tokens=500,
)
print(resp.choices[0].message.content)
```

### Listar os modelos disponíveis

```bash
curl -H "Authorization: Bearer $DGX_API_KEY" http://localhost:8000/v1/models
```

### Em ferramentas prontas (Continue, Cursor, LangChain, LiteLLM...)

Configure como um provedor "OpenAI-compatible":
- **Base URL / API Base:** a do seu cenário (com `/v1`)
- **API key:** a chave do grupo
- **Model:** `nemotron-3-super` ou `gemma-4-31b`

## 5. Interface de chat no navegador (opcional)

Existe um frontend de teste (arquivo único `chat_ui/index.html` — peça ao operador) com
streaming, métricas de velocidade e seletor de modelo. Para usar:

```bash
# na pasta onde salvou o index.html:
python3 -m http.server 8787
# abra http://localhost:8787 no navegador
```

Na primeira visita, clique em **⚙ Config** e informe a Base URL do seu cenário e a chave da
API (ficam salvas no seu navegador). *Nota: se o operador lhe enviar o arquivo com a chave
já embutida, trate o arquivo como confidencial.*

## 6. Limites e boas práticas

| Item | Valor | Comentário |
|---|---|---|
| Contexto Nemotron | 262.144 tokens | prompts gigantes levam minutos de processamento antes do 1º token — use timeout generoso (≥ 10 min) |
| Contexto Gemma | 32.768 tokens | requisições acima disso retornam erro 400 |
| Concorrência | ~30 reqs (Nemotron) / ~8 reqs (Gemma) | o vLLM enfileira o excedente; várias pessoas podem usar ao mesmo tempo |
| Velocidade | ~13–16 tok/s | é um cluster de inferência local, não a nuvem — respostas longas levam ~1 min |

- **Prefira o Gemma** para perguntas rápidas e volume alto; **reserve o Nemotron** para
  raciocínio difícil e contextos longos. Isso distribui a carga entre as duas máquinas.
- Erros transitórios de conexão no Cenário B geralmente são o túnel reconectando — espere
  ~10 s e reenvie.

## 7. Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `401 Unauthorized` | Chave ausente/errada | Confira o header `Authorization: Bearer <chave>` (no SDK, o `api_key`) |
| Timeout no `/health` (Cenário A) | Sua sub-rede filtra a porta 8000 | Use o Cenário B (túnel) |
| Timeout no `/health` (Cenário B) | Túnel caiu ou não subiu | Verifique o terminal do túnel; recrie-o. Teste o SSH puro: `ssh -p 1232 nvidia@200.144.205.82 echo ok` |
| `Permission denied` no SSH | Sua chave SSH não foi autorizada | Reenvie sua `.pub` ao operador |
| `400` com "maximum context length" | Prompt maior que a janela do modelo | Reduza o prompt ou use o Nemotron (256K) |
| `404 model not found` | Nome do modelo errado | Use exatamente `nemotron-3-super` ou `gemma-4-31b` (confira em `/v1/models`) |
| 1ª resposta muito lenta | Compilação pós-recarga do modelo | Normal; as seguintes são rápidas |
| Resposta do Nemotron "cortada" | `max_tokens` baixo demais p/ modelo de raciocínio | Aumente para ≥ 1024 |

## 8. Para o operador (referência rápida)

- **Autorizar novo usuário (Cenário B):** adicionar a chave pública dele em
  `~/.ssh/authorized_keys` do usuário `nvidia` no spark1.
- **Chave da API:** `sudo cat /etc/vllm/api.env` no spark1.
- **Arquitetura:** 1 modelo por nó + router (`~/router.py` no spark1, porta 8000).
  Detalhes completos, comandos de operação e lições aprendidas: `PLANO.md` deste repositório
  e o "Guia de Uso — Cluster DGX Spark" (PDF da BITINNOV).
