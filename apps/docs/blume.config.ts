import { withPreset } from '@fdarian/blume-preset';
import { defineConfig } from 'blume';
import { posthog } from 'blume/analytics';

const ph = process.env.POSTHOG_KEY;

export default defineConfig(
	withPreset({
		title: 'ai-effect',
		navigation: { repo: false },
		...(ph ? { analytics: [posthog({ key: ph })] } : {}),
	}),
);
