#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {
using Mask = uint32_t;
bool read_masks(PyObject* value, std::vector<Mask>& out, Py_ssize_t limit=25) {
    PyObject* seq=PySequence_Fast(value,"expected a sequence of integer masks");
    if (!seq) return false;
    Py_ssize_t n=PySequence_Fast_GET_SIZE(seq);
    if (n>limit) {Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"too many inventory variables");return false;}
    for(Py_ssize_t i=0;i<n;++i) {
        unsigned long v=PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(seq,i));
        if(PyErr_Occurred()) {Py_DECREF(seq);return false;}
        out.push_back(static_cast<Mask>(v));
    }
    Py_DECREF(seq);return true;
}
std::vector<Mask> edges_for(const std::vector<Mask>& domains,const std::vector<Mask>& slots) {
    std::vector<Mask> edges(domains.size());
    for(size_t i=0;i<domains.size();++i)
        for(int j=0;j<25;++j) if(domains[i]&slots[j]) edges[i]|=Mask(1)<<j;
    return edges;
}
bool augment(int i,const std::vector<Mask>& edges,std::array<int,25>& right,Mask& seen) {
    Mask available=edges[i]&~seen;
    while(available) {
        Mask bit=available&-available;available-=bit;
        if(seen&bit) continue;
        seen|=bit;int j=__builtin_ctz(bit);
        if(right[j]<0||augment(right[j],edges,right,seen)) {right[j]=i;return true;}
    }
    return false;
}
bool matching(const std::vector<Mask>& edges,std::array<int,25>& right) {
    right.fill(-1);std::vector<int> order(edges.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return __builtin_popcount(edges[a])<__builtin_popcount(edges[b]);});
    for(int i:order) {Mask seen=0;if(!augment(i,edges,right,seen)) return false;}
    return true;
}
struct SCC {
    std::array<uint64_t,50> graph{};
    std::array<int,50> index,low,component,stack;
    uint64_t active=0;int serial=0,top=0;
    SCC() {index.fill(-1);component.fill(-1);}
    void visit(int v) {
        index[v]=low[v]=serial++;stack[top++]=v;active|=uint64_t(1)<<v;
        uint64_t edges=graph[v];
        while(edges) {
            int w=__builtin_ctzll(edges);edges&=edges-1;
            if(index[w]<0) {visit(w);low[v]=std::min(low[v],low[w]);}
            else if(active&(uint64_t(1)<<w)) low[v]=std::min(low[v],index[w]);
        }
        if(low[v]==index[v]) while(true) {
            int w=stack[--top];active&=~(uint64_t(1)<<w);component[w]=v;if(w==v)break;
        }
    }
};
PyObject* supported(PyObject*,PyObject* args) {
    PyObject *domains_arg,*slots_arg;
    if(!PyArg_ParseTuple(args,"OO",&domains_arg,&slots_arg))return nullptr;
    std::vector<Mask> domains,slots;
    if(!read_masks(domains_arg,domains)||!read_masks(slots_arg,slots))return nullptr;
    if(domains.size()!=25||slots.size()!=25) {PyErr_SetString(PyExc_ValueError,"inventory requires exactly 25 live/removed variables");return nullptr;}
    auto edges=edges_for(domains,slots);std::array<int,25> right;
    if(!matching(edges,right)) {PyErr_SetString(PyExc_ValueError,"public inventory constraints have no feasible assignment");return nullptr;}
    SCC scc;
    for(int i=0;i<25;++i)for(int j=0;j<25;++j)if(edges[i]&(Mask(1)<<j)) {
        if(right[j]==i)scc.graph[25+j]|=uint64_t(1)<<i;
        else scc.graph[i]|=uint64_t(1)<<(25+j);
    }
    for(int v=0;v<50;++v)if(scc.index[v]<0)scc.visit(v);
    PyObject* result=PyTuple_New(25);
    for(int i=0;i<25;++i) {
        Mask bits=0;
        for(int j=0;j<25;++j)if((edges[i]&(Mask(1)<<j))&&(right[j]==i||scc.component[i]==scc.component[25+j]))bits|=slots[j];
        PyTuple_SET_ITEM(result,i,PyLong_FromUnsignedLong(bits));
    }
    return result;
}
PyObject* casualties(PyObject*,PyObject* args) {
    PyObject *live_arg,*dead_arg,*slots_arg,*types_arg;
    if(!PyArg_ParseTuple(args,"OOOO",&live_arg,&dead_arg,&slots_arg,&types_arg))return nullptr;
    std::vector<Mask> live,dead,slots,types;
    if(!read_masks(live_arg,live)||!read_masks(dead_arg,dead)||!read_masks(slots_arg,slots)||!read_masks(types_arg,types))return nullptr;
    if(slots.size()!=25||live.size()+dead.size()>25) {PyErr_SetString(PyExc_ValueError,"invalid inventory size");return nullptr;}
    std::vector<Mask> all=live;all.insert(all.end(),dead.begin(),dead.end());
    auto edges=edges_for(all,slots);PyObject* result=PyTuple_New(types.size());
    for(size_t t=0;t<types.size();++t) {
        std::vector<Mask> copies;int options=0;bool dead_option=false;
        for(int j=0;j<25;++j)if(slots[j]==types[t])copies.push_back(Mask(1)<<j);
        for(Mask d:live)options+=bool(d&types[t]);
        for(Mask d:dead)dead_option|=bool(d&types[t]);
        int maximum=std::min(int(copies.size()),options);
        if(maximum>1&&dead_option)while(maximum>1) {
            Mask reserved=0;for(int k=0;k<maximum;++k)reserved|=copies[k];
            auto choices=edges;for(size_t k=live.size();k<choices.size();++k)choices[k]&=~reserved;
            std::array<int,25> right;if(matching(choices,right))break;--maximum;
        }
        PyTuple_SET_ITEM(result,t,PyLong_FromLong(dead.empty()?0:int(copies.size())-maximum));
    }
    return result;
}

struct Geometry {
    std::array<std::vector<int>,60> neighbors, rails;
    std::array<std::array<int,60>,60> departure, arrival;
    std::array<int,60> flags{};
};
void delete_geometry(PyObject* capsule) {
    delete static_cast<Geometry*>(PyCapsule_GetPointer(capsule,"junqi.Geometry60"));
}
PyObject* geometry60(PyObject*,PyObject* args) {
    PyObject *neighbors,*rails,*directions,*flags;
    if(!PyArg_ParseTuple(args,"OOOO",&neighbors,&rails,&directions,&flags))return nullptr;
    if(!PyTuple_Check(neighbors)||PyTuple_Size(neighbors)!=60||!PyTuple_Check(rails)||PyTuple_Size(rails)!=60||!PyTuple_Check(flags)||PyTuple_Size(flags)!=60) {
        PyErr_SetString(PyExc_ValueError,"expected immutable 60-point geometry");return nullptr;
    }
    auto* g=new Geometry();
    for(int i=0;i<60;++i) {
        g->departure[i].fill(-1);g->arrival[i].fill(-1);
        g->flags[i]=int(PyLong_AsLong(PyTuple_GetItem(flags,i)));
        for(auto entry:{std::make_pair(neighbors,&g->neighbors[i]),std::make_pair(rails,&g->rails[i])}) {
            PyObject* row=PyTuple_GetItem(entry.first,i);
            if(!PyTuple_Check(row)){delete g;PyErr_SetString(PyExc_ValueError,"expected neighbor tuples");return nullptr;}
            for(Py_ssize_t j=0;j<PyTuple_Size(row);++j) {
                long value=PyLong_AsLong(PyTuple_GetItem(row,j));
                if(value<0||value>=60){delete g;PyErr_SetString(PyExc_ValueError,"invalid geometry index");return nullptr;}
                entry.second->push_back(int(value));
            }
        }
    }
    PyObject* seq=PySequence_Fast(directions,"expected rail tangents");
    if(!seq){delete g;return nullptr;}
    for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq);++i) {
        PyObject* row=PySequence_Fast_GET_ITEM(seq,i);int a,b,d,e;
        if(!PyArg_ParseTuple(row,"iiii",&a,&b,&d,&e)||a<0||a>=60||b<0||b>=60||d<0||d>=4||e<0||e>=4) {
            Py_DECREF(seq);delete g;if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"invalid rail tangent");return nullptr;
        }
        g->departure[a][b]=d;g->arrival[a][b]=e;
    }
    Py_DECREF(seq);
    if(PyErr_Occurred()){delete g;return nullptr;}
    return PyCapsule_New(g,"junqi.Geometry60",delete_geometry);
}
uint64_t rail_targets(const Geometry& g,int start,uint64_t occupied,bool engineer) {
    uint64_t reachable=0;std::array<std::array<bool,5>,60> visited{};
    std::array<std::pair<int,int>,301> queue;int head=0,tail=1;
    queue[0]={start,4};visited[start][4]=true;
    while(head<tail) {
        auto [current,direction]=queue[head++];
        for(int neighbor:g.rails[current]) {
            if(neighbor==start)continue;
            int departure=g.departure[current][neighbor],arrival=g.arrival[current][neighbor];
            if(!engineer&&(departure<0||(direction!=4&&departure!=direction)))continue;
            reachable|=uint64_t(1)<<neighbor;
            int next=engineer?4:arrival;
            if(occupied&(uint64_t(1)<<neighbor)||visited[neighbor][next])continue;
            visited[neighbor][next]=true;queue[tail++]={neighbor,next};
        }
    }
    return reachable;
}
PyObject* railway60(PyObject*,PyObject* args) {
    PyObject* capsule;int start,engineer;unsigned long long occupied;
    if(!PyArg_ParseTuple(args,"OiKp",&capsule,&start,&occupied,&engineer))return nullptr;
    auto* g=static_cast<Geometry*>(PyCapsule_GetPointer(capsule,"junqi.Geometry60"));if(!g)return nullptr;
    if(start<0||start>=60){PyErr_SetString(PyExc_ValueError,"invalid start");return nullptr;}
    uint64_t targets;
    Py_BEGIN_ALLOW_THREADS
    targets=rail_targets(*g,start,occupied,engineer);
    Py_END_ALLOW_THREADS
    PyObject* result=PyTuple_New(__builtin_popcountll(targets));int j=0;
    while(targets){int i=__builtin_ctzll(targets);targets&=targets-1;PyTuple_SET_ITEM(result,j++,PyLong_FromLong(i));}
    return result;
}
PyObject* legal60(PyObject*,PyObject* args) {
    PyObject* capsule;Py_buffer encoded;int player,passes,eliminate;
    if(!PyArg_ParseTuple(args,"Oy*ipp",&capsule,&encoded,&player,&passes,&eliminate))return nullptr;
    auto* g=static_cast<Geometry*>(PyCapsule_GetPointer(capsule,"junqi.Geometry60"));
    if(!g||encoded.len!=60||player<0||player>1){PyBuffer_Release(&encoded);if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"invalid board");return nullptr;}
    std::array<unsigned char,60> board;std::copy_n(static_cast<unsigned char*>(encoded.buf),60,board.begin());PyBuffer_Release(&encoded);
    std::vector<std::pair<int,int>> actions;
    Py_BEGIN_ALLOW_THREADS
    uint64_t occupied=0;for(int i=0;i<60;++i)if(board[i])occupied|=uint64_t(1)<<i;
    for(int i=0;i<60;++i)if((board[i]&3)==player+1&&(board[i]&4)&&!(g->flags[i]&2)) {
        uint64_t targets=rail_targets(*g,i,occupied,board[i]&8);
        for(int j:g->neighbors[i])targets|=uint64_t(1)<<j;
        while(targets) {
            int j=__builtin_ctzll(targets);targets&=targets-1;
            if(!board[j]||((board[j]&3)!=player+1&&!(g->flags[j]&1)))actions.emplace_back(i,j);
        }
    }
    if(passes&&(!actions.empty()||!eliminate))actions.insert(actions.begin(),{0,0});
    Py_END_ALLOW_THREADS
    PyObject* result=PyTuple_New(actions.size());
    for(size_t i=0;i<actions.size();++i)PyTuple_SET_ITEM(result,i,Py_BuildValue("ii",actions[i].first,actions[i].second));
    return result;
}


PyMethodDef methods[]={{"geometry60",geometry60,METH_VARARGS,"Cache immutable integer geometry."},{"railway60",railway60,METH_VARARGS,"Exact rail reachability."},{"legal60",legal60,METH_VARARGS,"Stable-order legal actions."},{"supported",supported,METH_VARARGS,"Public-domain matching and SCC support."},{"casualties",casualties,METH_VARARGS,"Minimum guaranteed public casualties."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_inventory_native",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__inventory_native(){return PyModule_Create(&module);}
