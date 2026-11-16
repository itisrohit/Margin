FROM oven/bun:1.2.16 AS dependencies

WORKDIR /app
COPY package.json bun.lock ./
COPY web/package.json ./web/package.json
RUN bun install --frozen-lockfile

FROM node:22-bookworm-slim

WORKDIR /app/web
COPY --from=dependencies /app/node_modules /app/node_modules
EXPOSE 5173
CMD ["node", "/app/node_modules/vite/bin/vite.js", "--host", "0.0.0.0"]
