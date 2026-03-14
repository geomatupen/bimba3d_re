import axios from "axios";

const hostname = typeof window !== 'undefined' ? window.location.hostname : '';
const port = typeof window !== 'undefined' ? window.location.port : '';
const protocol = typeof window !== 'undefined' ? window.location.protocol : 'http:';

const isViteDevPort = port === '5173' || port === '5174';

const baseURL = hostname === 'app.bimba3d.com'
  ? 'https://backend.bimba3d.com'
  : isViteDevPort
    ? 'http://localhost:8005'
    : `${protocol}//${hostname}${port ? `:${port}` : ''}`;

export const api = axios.create({
  baseURL,
});
