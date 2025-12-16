# ==== FRONTEND: React + Nginx ====
# 1️⃣ Build Stage
FROM node:22-alpine AS build

# Set working directory
WORKDIR /app

# Copy package files & install deps
COPY package*.json ./
RUN npm install

# Copy all source code
COPY . .

# Build the production files
RUN npm run build

# 2️⃣ Serve Stage
FROM nginx:alpine

# Remove default nginx page
RUN rm -rf /usr/share/nginx/html/*

# Copy built React files to nginx directory
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
