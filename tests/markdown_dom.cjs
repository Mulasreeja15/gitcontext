// Minimal DOM for executing the production renderer without a browser or network.
const fs = require('node:fs');
class Node {
  constructor(tag, text = '') { this.tag = tag; this.text = text; this.children = []; this.attrs = {}; }
  append(...nodes) {
    for (const node of nodes) {
      if (node.tag === '#fragment') this.append(...node.children);
      else this.children.push(node);
    }
  }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  set textContent(value) { this.children = []; this.text = value; }
  get textContent() { return this.text + this.children.map(node => node.textContent).join(''); }
  set innerHTML(_) { throw new Error('Unsafe HTML insertion'); }
  setAttribute(name, value) { this.attrs[name] = value; }
  addEventListener() {}
}
const answer = new Node('div');
const document = {
  baseURI: 'http://localhost:8000/',
  createElement: tag => new Node(tag),
  createTextNode: text => new Node('#text', text),
  createDocumentFragment: () => new Node('#fragment'),
};
const $ = () => answer, sourceCount = 4, focusSource = () => {};
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
eval(input.renderer + '\nmarkdown(input.sample);');
function serialize(node) {
  return {tag: node.tag, text: node.textContent, attrs: node.attrs,
    href: node.href, rel: node.rel, children: node.children.map(serialize)};
}
process.stdout.write(JSON.stringify(serialize(answer)));
