import { Router } from 'itty-router';
import YAML from 'yaml';

// now let's create a router (note the lack of "new")
const router = Router();

// GET collection index
router.get('/api/:branch/items', async ({ params }) => {
	console.log({
		branch: params.branch,
		event_type: 'COLLECTION_INDEX',
		timestamp: new Date().toISOString(),
	});
	const url = `https://raw.githubusercontent.com/jxnl/instructor/${params.branch}/mkdocs.yml?raw=true`;
	const mkdoc_yml = await fetch(url).then((res) => res.text());
	var mkdocs = YAML.parse(mkdoc_yml);
	const cookbooks = mkdocs.nav
		?.filter((obj: Map<string, string>) => 'Hub' in obj)[0]
		.Hub.map((obj: any, index: number) => {
			const [name, path] = Object.entries(obj)[0];
			// Extract slug by getting the substring after the last '/'
			// @ts-ignore
			const slug = path.substring(path.lastIndexOf('/') + 1, path.lastIndexOf('.'));
			return { id: index, name, path, slug };
		})
		.filter(({ slug }: any) => slug !== 'index');

	return new Response(JSON.stringify(cookbooks), {
		headers: {
			'content-type': 'application/json',
		},
	});
});

// GET content
router.get('/api/:branch/items/:slug/md', async ({ params }) => {
	console.log({
		branch: params.branch,
		slug: params.slug,
		event_type: 'CONTENT_MARKDOWN',
		timestamp: new Date().toISOString(),
	});
	const raw_content = `https://raw.githubusercontent.com/jxnl/instructor/${params.branch}/docs/hub/${params.slug}.md?raw=true`;
	const content = await fetch(raw_content).then((res) => res.text());

	return new Response(content, {
		headers: {
			'content-type': 'text/plain',
		},
	});
});

// GET content python
router.get('/api/:branch/items/:slug/py', async ({ params }) => {
	console.log({
		branch: params.branch,
		slug: params.slug,
		event_type: 'CONTENT_PYTHON',
		timestamp: new Date().toISOString(),
	});
	const raw_content = `https://raw.githubusercontent.com/jxnl/instructor/${params.branch}/docs/hub/${params.slug}.md?raw=true`;
	const content = await fetch(raw_content).then((res) => res.text());

	// Extract all Python code blocks from within ```py or ```python blocks in the markdown
	const python_codes = content.match(/(?<=```(?:py|python)\n)[\s\S]+?(?=\n```)/g);

	if (python_codes === null) {
		return new Response('No Python code found in this document.', {
			headers: {
				'content-type': 'text/plain',
			},
		});
	}

	if (python_codes.length === 0) {
		return new Response('No Python code found in this document.', {
			headers: {
				'content-type': 'text/plain',
			},
		});
	}

	const python_code = python_codes.join('\n\n');

	return new Response(python_code, {
		headers: {
			'content-type': 'text/plain',
		},
	});
});

// 404 for everything else
router.all('*', () => new Response('Not Found.', { status: 404 }));

export default router;
