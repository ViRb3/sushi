// The chat page's thinking-effort menu, driven through the page's own functions without a browser or model.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('src/webui/index.html', 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
const slice = (from, to) => {
  const start = script.indexOf(from);
  return script.slice(start, script.indexOf(to, start));
};

// A browser select holds only a value one of its options carries, and starts on its first option.
function fakeSelect() {
  const select = { dataset: {}, disabled: false, options: [], current: '' };
  Object.defineProperty(select, 'textContent', { set() { select.options = []; select.current = ''; } });
  select.appendChild = (option) => {
    select.options.push(option);
    if (select.options.length === 1) select.current = option.value;
  };
  Object.defineProperty(select, 'value', {
    get: () => select.current,
    set: (v) => { select.current = select.options.some((o) => o.value === v) ? v : ''; },
  });
  return select;
}

const select = fakeSelect();
const label = { textContent: '' };
const store = {};
let model = null;
const page = vm.createContext({
  $: (id) => ({ effortSelect: select, composerEffortName: label })[id],
  el: (tag, className, text) => ({ tag, text, value: '' }),
  readStore: (key) => store[key] ?? null,
  selectedModel: () => model,
});
vm.runInContext(slice('function renderEffortOptions(', 'function renderModelStatus('), page);
const show = (m) => { model = m; page.renderEffortOptions(); return select.value; };

const models = {
  qwen: { id: 'qwen', reasoning_efforts: ['off', 'low', 'medium', 'xhigh'], default_reasoning_effort: 'off' },
  mimo: { id: 'mimo', reasoning_efforts: ['off', 'on'], default_reasoning_effort: 'on' },
  glm: { id: 'glm', reasoning_efforts: ['low', 'high', 'max'], default_reasoning_effort: 'high' },
};
for (const m of Object.values(models)) {
  assert.equal(show(m), m.default_reasoning_effort, `${m.id} starts on its own default`);
  assert.deepEqual(select.options.map((o) => o.value), m.reasoning_efforts, `${m.id} lists exactly its own words`);
  assert.equal(label.textContent, m.default_reasoning_effort, `${m.id} label names the real word`);
}

// A stored choice holds on a model that takes it; elsewhere that model's default applies.
store['sushi-effort'] = 'off';
assert.equal(show(models.mimo), 'off');
assert.equal(show(models.glm), 'high');
assert.equal(label.textContent, 'high');
assert.equal(show(models.qwen), 'off');
store['sushi-effort'] = 'default';
assert.equal(show(models.mimo), 'on');

// Every request names the selected effort.
const request = vm.createContext({ $: (id) => (id === 'effortSelect' ? select : { value: '' }) });
vm.runInContext(slice('function wireMessages(', 'function createStreamingView('), request);
assert.equal(request.buildRequest('mimo', []).reasoning_effort, 'on');
show(models.qwen);
assert.equal(request.buildRequest('qwen', []).reasoning_effort, 'off');
console.log('Web UI effort menu: passed');
