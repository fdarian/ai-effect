import { withPreset } from '@fdarian/blume-preset';
import { defineConfig } from 'blume';

export default defineConfig(
	withPreset({ title: 'ai-effect', navigation: { repo: false } }),
);
