import { defineComponents } from 'blume';

export default defineComponents({
	layout: {
		Sidebar: './src/components/Sidebar.astro',
		TableOfContents: {
			component: '@fdarian/blume-preset/chapters.tsx',
			client: 'load',
		},
	},
});
