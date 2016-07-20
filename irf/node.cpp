/* Copyright 2012, Igalia S.L.
 * Author Carlos Guerreiro cguerreiro@igalia.com
 * Licensed under the MIT license */

#include <cstdio>
#include <sstream>

#include <v8.h>
#include <node.h>
#include <node_buffer.h>
#include "nan.h"
#include "randomForest.h"

using namespace v8;
using namespace node;
using namespace std;
using namespace IncrementalRandomForest;

static Nan::Persistent<Function> constructor;


class IRF: public ObjectWrap {
private:
  Forest* f;

  static void setFeatures(Sample* s, Local<Object>& features) {
    //Local<Array> featureNames = features->GetOwnPropertyNames();
    Local<Array> featureNames = Nan::GetOwnPropertyNames(features).ToLocalChecked();
    int featureCount = featureNames->Length();
    int i;
    for(i = 0; i < featureCount; ++i) {
      // FIXME: verify that this is an integer
      Local<Integer> n = featureNames->Get(i)->ToInteger();
      // FIXME: verify that this is a number
      Local<Number> v = features->Get(n->Value())->ToNumber();
      s->xCodes[n->Value()] = v->Value();
    }
  }

  static void getFeatures(Sample* s, Local<Object>& features) {
    map<int, float>::const_iterator it;
    char key[32];
    for(it = s->xCodes.begin(); it != s->xCodes.end(); ++it) {
      snprintf(key, 31, "%d", it->first);
      features->Set(Nan::New<String>(key).ToLocalChecked(), Nan::New<Number>(it->second));
    }
  }

public:
  IRF(uint32_t count) : ObjectWrap() {
    f = IncrementalRandomForest::create(count);
  }

  IRF(Forest* withF) : ObjectWrap(), f(withF) {
  }

  ~IRF() {
    IncrementalRandomForest::destroy(f);
  }

  static NAN_MODULE_INIT(init) {
    Local<FunctionTemplate> tpl = Nan::New<FunctionTemplate>(IRF::New);

    tpl->SetClassName(Nan::New("IRF").ToLocalChecked());
    tpl->InstanceTemplate()->SetInternalFieldCount(1);

    Nan::SetPrototypeMethod(tpl, "add", add);
    Nan::SetPrototypeMethod(tpl, "remove", remove);
    Nan::SetPrototypeMethod(tpl, "classify", classify);
    Nan::SetPrototypeMethod(tpl, "classifyPartial", classifyPartial);
    Nan::SetPrototypeMethod(tpl, "asJSON", asJSON);
    Nan::SetPrototypeMethod(tpl, "statsJSON", statsJSON);
    Nan::SetPrototypeMethod(tpl, "each", each);
    Nan::SetPrototypeMethod(tpl, "commit", commit);
    Nan::SetPrototypeMethod(tpl, "toBuffer", toBuffer);

    constructor.Reset(Nan::GetFunction(tpl).ToLocalChecked());
    Nan::Set(target, Nan::New("IRF").ToLocalChecked(),
      Nan::GetFunction(tpl).ToLocalChecked());
  }

  static NAN_METHOD(fromBuffer) {
    if(info.Length() != 1) {
      return Nan::ThrowError("add takes 3 arguments");
    }

    if(!Buffer::HasInstance(info[0]))
      return Nan::ThrowError("argument must be a Buffer");

    Local<Object> o = info[0]->ToObject();

    cerr << Buffer::Length(o) << endl;

    stringstream ss(Buffer::Data(o));

    IRF* ih = new IRF(load(ss));
    ih->Wrap(info.This());
    info.GetReturnValue().Set(info.This());
  }

  static NAN_METHOD(New) {
    Nan::HandleScope scope;

    if (!info.IsConstructCall()) {
      return Nan::ThrowTypeError("Use the new operator to create instances of this object.");
    }

    IRF* ih;
    if(info.Length() >= 1) {
      if(info[0]->IsNumber()) {
        uint32_t count = info[0]->ToInteger()->Value();
        ih = new IRF(count);
      } else if(Buffer::HasInstance(info[0])) {
        Local<Object> o = info[0]->ToObject();
        stringstream ss(Buffer::Data(o));
        ih = new IRF(load(ss));
      } else {
        return Nan::ThrowError("argument 1 must be a number (number of trees) or a Buffer (to create from)");
      }
    } else
      ih = new IRF(1);

    ih->Wrap(info.This());
    info.GetReturnValue().Set(info.This());
  }

  static NAN_METHOD(add) {
    Nan::HandleScope scope;

    if(info.Length() != 3) {
      return Nan::ThrowError("add takes 3 arguments");
    }

    Local<String> suid = Nan::To<String>(info[0]).ToLocalChecked();
    if(suid.IsEmpty())
      return Nan::ThrowError("argument 1 must be a string");

    if(!info[1]->IsObject())
      return Nan::ThrowError("argument 2 must be a object");
    Local<Object> features = Nan::To<Object>(info[1]).ToLocalChecked();

    if(!info[2]->IsNumber())
      return Nan::ThrowError("argument 3 must be a number");
    Local<Number> y = Nan::To<Number>(info[2]).ToLocalChecked();

    IRF* ih = ObjectWrap::Unwrap<IRF>(info.This());
    Sample* s = new Sample();
    s->suid = *Nan::Utf8String(suid);
    s->y = y->Value();
    setFeatures(s, features);
    info.GetReturnValue().Set(Nan::New<Boolean>(IncrementalRandomForest::add(ih->f, s)));
  }

  static NAN_METHOD(remove) {
    Nan::HandleScope scope;

    if(info.Length() != 1) {
      return Nan::ThrowError("remove takes 1 argument");
    }

    Local<String> suid = Nan::To<String>(info[0]).ToLocalChecked();
    if(suid.IsEmpty())
      return Nan::ThrowError("argument 1 must be a string");

    IRF* ih = ObjectWrap::Unwrap<IRF>(info.This());
    info.GetReturnValue().Set(IncrementalRandomForest::remove(ih->f, *Nan::Utf8String(suid)));
  }

  static NAN_METHOD(classify) {
    Nan::HandleScope scope;

    if(info.Length() != 1) {
      return Nan::ThrowError("classify takes 1 argument");
    }

    if(!info[0]->IsObject())
      return Nan::ThrowError("argument 1 must be a object");
    Local<Object> features = Nan::To<Object>(info[0]).ToLocalChecked();

    IRF* ih = ObjectWrap::Unwrap<IRF>(info.This());

    IncrementalRandomForest::Sample s;
    setFeatures(&s, features);

    info.GetReturnValue().Set(Nan::New<Number>(IncrementalRandomForest::classify(ih->f, &s)));
  }

  static NAN_METHOD(classifyPartial) {
    Nan::HandleScope scope;

    if(info.Length() != 2) {
      return Nan::ThrowError("classifyPartial takes 2 argument");
    }

    if(!info[0]->IsObject())
      return Nan::ThrowError("argument 1 must be a object");
    Local<Object> features = Nan::To<Object>(info[0]).ToLocalChecked();


    if(!info[1]->IsNumber())
      return Nan::ThrowError("argument 2 must be a number");
    Local<Number> nTrees = Nan::To<Number>(info[1]).ToLocalChecked();

    IRF* ih = ObjectWrap::Unwrap<IRF>(info.This());

    IncrementalRandomForest::Sample s;
    setFeatures(&s, features);
    
    info.GetReturnValue().Set(Nan::New<Number>(IncrementalRandomForest::classifyPartial(ih->f, &s, nTrees->Value())));
  }

  static NAN_METHOD(asJSON) {
    Nan::HandleScope scope;

    if(info.Length() != 0) {
      return Nan::ThrowError("toJSON takes 0 arguments");
    }

    IRF* ih = ObjectWrap::Unwrap<IRF>(info.This());

    stringstream ss;
    IncrementalRandomForest::asJSON(ih->f, ss);
    ss.flush();

    info.GetReturnValue().Set(Nan::New<String>(ss.str().c_str()).ToLocalChecked());
  }

  static NAN_METHOD(statsJSON) {
    Nan::HandleScope scope;

    if(info.Length() != 0) {
      return Nan::ThrowError("statsJSON takes 0 arguments");
    }

    IRF* ih = ObjectWrap::Unwrap<IRF>(info.This());

    stringstream ss;
    IncrementalRandomForest::statsJSON(ih->f, ss);
    ss.flush();

    info.GetReturnValue().Set(Nan::New<String>(ss.str().c_str()).ToLocalChecked());
  }

  static NAN_METHOD(each) {
    Nan::HandleScope scope;

    if(info.Length() != 1) {
      return Nan::ThrowError("each takes 1 argument");
    }
    if (!info[0]->IsFunction()) {
      return Nan::ThrowError("argument must be a callback function");
    }
    // There's no ToFunction(), use a Cast instead.
    Local<Function> callback = Local<Function>::Cast(info[0]);

    Local<Value> k = Nan::Undefined();
    Local<Value> v = Nan::Undefined();

    const unsigned argc = 3;
    Local<Value> argv[argc] = { v };

    IRF* ih = ObjectWrap::Unwrap<IRF>(info.This());

    SampleWalker* walker = getSamples(ih->f);

    Local<Object> globalObj = Nan::GetCurrentContext()->Global();
    Local<Function> objectConstructor = Local<Function>::Cast(globalObj->Get(Nan::New("Object").ToLocalChecked()));

    while(walker->stillSome()) {
      Sample* s = walker->get();
      argv[0] = Nan::New<String>(s->suid.c_str()).ToLocalChecked();
      Local<Object> features = Nan::New<Object>();
      getFeatures(s, features);
      argv[1] = features;
      argv[2] = Nan::New<Number>(s->y);
      TryCatch tc;
      Local<Value> ret = callback->Call(Nan::GetCurrentContext()->Global(), argc, argv);
      if(ret.IsEmpty() || ret->IsFalse())
        break;
    }

    delete walker;

    info.GetReturnValue().SetUndefined();
  }

  static NAN_METHOD(commit) {
    Nan::HandleScope scope;

    if(info.Length() != 0) {
      return Nan::ThrowError("commit takes 0 arguments");
    }

    IRF* ih = ObjectWrap::Unwrap<IRF>(info.This());

    IncrementalRandomForest::commit(ih->f);

    info.GetReturnValue().SetUndefined();  
  }

  static NAN_METHOD(toBuffer) {
    Nan::HandleScope scope;

    if(info.Length() != 0) {
      return Nan::ThrowError("save takes 0 arguments");
    }

    IRF* ih = ObjectWrap::Unwrap<IRF>(info.This());
    stringstream ss(stringstream::out | stringstream::binary);
    save(ih->f, ss);
    ss.flush();

    Local<Object> out = Nan::CopyBuffer(const_cast<char*>(ss.str().c_str()), ss.tellp()).ToLocalChecked();
    info.GetReturnValue().Set(out);
  }
};


NODE_MODULE(irf, IRF::init);
