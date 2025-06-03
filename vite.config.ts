import { defineConfig } from 'vite';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  build: {
    lib: {
      entry: './src/trainDataset.ts',
      name: 'action-recog',
    },
    outDir: 'dist',
    emptyOutDir: true,
  },
  plugins: [tsconfigPaths()],
});
