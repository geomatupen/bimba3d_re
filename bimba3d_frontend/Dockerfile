FROM node:20-alpine AS builder

WORKDIR /app


COPY package*.json ./
RUN npm ci
# Patch maplibre-gl.d.ts to fix TypeScript build error
RUN sed -i '449s/);/}/' node_modules/maplibre-gl/dist/maplibre-gl.d.ts

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html

# Configure nginx for SPA
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
