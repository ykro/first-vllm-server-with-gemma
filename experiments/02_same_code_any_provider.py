"""02. Same app code, different server: the OpenAI SDK pointed at vLLM.

The application function `summarize_ticket` knows nothing about vLLM. The only line that
decides where the model runs is the one that builds the client (base_url). It also shows
that with temperature 0 and a fixed seed the same question gives the same answer twice.

    python experiments/02_same_code_any_provider.py [--mock]
"""

from __future__ import annotations

from openai import OpenAI

from vllm_lab import MODEL, cli, make_client

TICKET = (
    "Cliente: compré unos audífonos el lunes, llegaron el jueves con la caja abierta y "
    "el lado izquierdo no suena. Ya probé con otro teléfono. Quiero cambio, no reembolso."
)


def summarize_ticket(client: OpenAI, ticket: str) -> str:
    """Plain application code. Works with any OpenAI-compatible server."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Resume tickets de soporte en una línea: problema y "
                                          "lo que pide el cliente."},
            {"role": "user", "content": ticket},
        ],
        max_tokens=60,
        temperature=0,
        seed=7,
    )
    return (resp.choices[0].message.content or "").strip()


def run(base_url: str) -> dict:
    client = make_client(base_url)          # <- the only vLLM-specific line
    first = summarize_ticket(client, TICKET)
    second = summarize_ticket(client, TICKET)
    return {"base_url": base_url, "summary": first, "repeatable": first == second}


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    cli.show("summarize_ticket() against vLLM", run(cli.resolve(args)))


if __name__ == "__main__":
    main()
