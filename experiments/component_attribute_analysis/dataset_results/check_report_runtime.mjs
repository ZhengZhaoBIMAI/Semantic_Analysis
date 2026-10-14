import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const html=fs.readFileSync(new URL('./dataset_semantic_report.html',import.meta.url),'utf8');
const scripts=[...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].map(x=>x[1]);
assert.equal(scripts.length,2);
const data=JSON.parse(scripts[0]);
const elements=new Map();
class Element {
  constructor(id){this.id=id;this.value='';this.checked=false;this.textContent='';this.listeners={};this.classes=new Set();this.classList={add:x=>this.classes.add(x),remove:x=>this.classes.delete(x)};}
  set innerHTML(v){this.html=v;const m=v.match(/<option(?: value="([^"]*)")?[^>]*>(.*?)<\/option>/);if(m)this.value=m[1]??m[2];}
  get innerHTML(){return this.html??'';}
  addEventListener(event,fn){this.listeners[event]=fn;}
  click(){if(this.onclick)this.onclick();}
}
const get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
get('data').textContent=scripts[0];get('minCoverage').value='0';
const tabIds=[...html.matchAll(/data-tab="([^"]+)"/g)].map(x=>x[1]);
const buttons=tabIds.map(id=>{const e=new Element('button:'+id);e.dataset={tab:id};return e;});
let downloaded='';
const document={getElementById:get,querySelectorAll:q=>q==='.tabs button'?buttons:[...buttons,...tabIds.map(get)],createElement:()=>new Element('download')};
const context=vm.createContext({document,console,Blob:class{constructor(parts){this.parts=parts;}},URL:{createObjectURL:b=>{downloaded=b.parts.join('');return 'test:download';},revokeObjectURL:()=>{}},setTimeout:fn=>fn()});
vm.runInContext(scripts[1],context,{timeout:30000});
assert.match(get('fieldCount').textContent,/5,392/);
get('fieldClass').value='IfcWall';get('fieldSearch').value='Pset_WallCommon.IsExternal';get('fieldSearch').listeners.input();
assert.match(get('fieldCount').textContent,/2 条/);assert.match(get('catalogBody').innerHTML,/97\.4%/);
get('exportFields').click();assert.equal(downloaded.charCodeAt(0),0xfeff);assert.match(downloaded,/Pset_WallCommon.IsExternal/);
get('sourceClass').value='IfcWall';get('sourceSearch').value='Pset_WallCommon.ThermalTransmittance';get('sourceSearch').listeners.input();
assert.match(get('sourceBody').innerHTML,/12,189/);assert.match(get('sourceBody').innerHTML,/7,830/);
assert.equal([...get('sourceBody').innerHTML.matchAll(/<td(?:\s[^>]*)?>/g)].length,10*[...get('sourceBody').innerHTML.matchAll(/<tr>/g)].length);
assert(html.includes('50% 是本次分析选用的筛选线'));assert(!html.includes('__READING_GUIDE__'));
get('vendorFilter').value='Archicad';get('vendorFilter').listeners.input();assert.match(get('modelCount').textContent,/13 个模型/);
get('caseModel').value='model_6';get('caseModel').listeners.input();get('caseClass').value='IfcDoor';get('caseClass').listeners.input();
const example=data.examples.find(x=>x.model==='model_6'&&x.component_type==='IfcDoor');
assert(example);assert(JSON.stringify(JSON.parse(get('caseProperties').textContent))===JSON.stringify(example.properties),'representative case values must survive JSON display');
buttons[2].listeners.click();assert(get('sourcesTab').classes.has('active'));
assert.equal(data.current.node_count,data.original.audited_nodes);assert.equal(data.current.model_count,data.original.audited_models);
assert(data.original.errors.length===0);assert(html.includes('src="data:image/png;base64,'));
console.log('PASS: report initialization, field/source/model filters, CSV export, actual model examples, tab switching, embedded image, population agreement');
