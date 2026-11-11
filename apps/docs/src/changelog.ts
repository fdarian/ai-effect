import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { marked } from 'marked';

export type ChangelogSection = {
	label: string;
	items: string[];
};

export type ChangelogRelease = {
	version: string;
	date: string;
	sections: ChangelogSection[];
};

// Resolved from the cwd (apps/docs): the build bundles this module into .blume-verify/ or .blume/, so import.meta.url no longer points into the repo.
const changelogPath = resolve(
	process.cwd(),
	'../../packages/ai-effect/CHANGELOG.md',
);

const sectionLabels: Record<string, string> = {
	'Major Changes': 'Breaking',
	'Minor Changes': 'Added',
	'Patch Changes': 'Fixed',
};

function releaseDate(version: string) {
	// `-S` counts occurrences of the string, so this finds the commits that added the heading.
	const output = execFileSync(
		'git',
		['log', '-S', `## ${version}\n`, '--format=%cs', '--', changelogPath],
		{ encoding: 'utf8' },
	);
	const dates = output.trim().split('\n').filter(Boolean);
	const oldest = dates[dates.length - 1];
	if (oldest === undefined) {
		throw new Error(
			`Cannot date release ${version}: no commit in git history added "## ${version}" to ${changelogPath}. Fetch full history (git fetch --unshallow).`,
		);
	}
	return oldest;
}

function renderItem(lines: string[]) {
	const text = lines
		.map((line) => line.replace(/^ {2}/, ''))
		.join('\n')
		.replace(/^[0-9a-f]{7,}: /, '');
	return marked.parse(text, { async: false });
}

type RawRelease = {
	version: string;
	sections: { label: string; items: string[][] }[];
};

function parseChangelog(markdown: string) {
	const releases: RawRelease[] = [];
	let release: RawRelease | undefined;
	let items: string[][] | undefined;

	for (const line of markdown.split('\n')) {
		if (line.startsWith('## ')) {
			release = { version: line.slice(3).trim(), sections: [] };
			releases.push(release);
			items = undefined;
		} else if (line.startsWith('### ')) {
			if (release === undefined) {
				throw new Error(`Section "${line}" appears before any release`);
			}
			const heading = line.slice(4).trim();
			const label = sectionLabels[heading];
			if (label === undefined) {
				throw new Error(`Unknown changelog section "${heading}"`);
			}
			items = [];
			release.sections.push({ label, items });
		} else if (items !== undefined) {
			if (line.startsWith('- ')) {
				items.push([line.slice(2)]);
			} else if (line.trim() !== '') {
				const current = items[items.length - 1];
				if (current === undefined) {
					throw new Error(`Unexpected changelog line "${line}"`);
				}
				current.push(line);
			}
		}
	}
	return releases;
}

export function getChangelog(): ChangelogRelease[] {
	return parseChangelog(readFileSync(changelogPath, 'utf8')).map((release) => ({
		version: release.version,
		date: releaseDate(release.version),
		sections: release.sections.map((section) => ({
			label: section.label,
			items: section.items.map(renderItem),
		})),
	}));
}
