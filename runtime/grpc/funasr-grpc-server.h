#pragma once

#include <mutex>
#include <string>
#include <thread>

#include "funasrruntime.h"
#include "glog/logging.h"
#include "tclap/CmdLine.h"

#include "fst/compat.h"
#include "paraformer.grpc.pb.h"

using paraformer::ASR;
using paraformer::DecodeMode;
using paraformer::Request;
using paraformer::Response;
using paraformer::WavFormat;

class GrpcEngine {
public:
    GrpcEngine(grpc::ServerReaderWriter<Response, Request> *stream, std::shared_ptr<FUNASR_HANDLE> asr_handle);

    void operator()();

private:
    void DecodeThreadFunc();

    void OnSpeechStart();

    void OnSpeechData();

    void OnSpeechEnd();

    grpc::ServerReaderWriter<Response, Request> *stream_;
    std::shared_ptr<Request> request_;
    std::shared_ptr<Response> response_;
    std::shared_ptr<FUNASR_HANDLE> asr_handle_;
    std::shared_ptr<FUNASR_DEC_HANDLE> decode_handle_;
    std::string audio_buffer_;
    bool is_start_ = false;

    int sampling_rate_ = 16000;
    std::string encoding_;
    ASR_TYPE mode_ = ASR_OFFLINE;
    bool itn_ = false;
    std::shared_ptr<std::vector<std::vector<float>>> hotwords_embedding_ = nullptr;
};

class GrpcService final : public ASR::Service {
public:
    GrpcService(std::map<std::string, std::string> &model_path, int model_thread_num);
    grpc::Status Recognize(grpc::ServerContext *context, grpc::ServerReaderWriter<Response, Request> *stream) override;

private:
    std::map<std::string, std::string> model_path_;
    std::shared_ptr<FUNASR_HANDLE> asr_handle_;
    DISALLOW_COPY_AND_ASSIGN(GrpcService);
};
