# Docker Kullanimi

Bu kurulum proje klasoru disinda baska projeleri etkilemez.

## 1) Ilk kurulum

```bash
docker compose up -d --build
```

## 2) Log izleme

```bash
docker compose logs -f bot
```

## 3) Durdurma

```bash
docker compose down
```

## Notlar

- Container adi: `crypto-bot-bot`
- Compose proje adi: `crypto-bot-local`
- Ortam degiskenleri `.env` dosyasindan okunur.
- `logs/` ve `trades.db` host ile paylasimda oldugu icin veri kalicidir.
