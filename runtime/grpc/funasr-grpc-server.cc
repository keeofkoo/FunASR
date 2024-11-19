#include <chrono>
#include <exception>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "com-define.h"
#include "funasr-grpc-server.h"
#include "funasrruntime.h"
#include "grpcpp/ext/proto_server_reflection_plugin.h"
#include "grpcpp/health_check_service_interface.h"
#include "grpcpp/server_builder.h"
#include "tclap/ValueArg.h"
#include "util.h"

std::unordered_map<std::string, int> hws_map_;
int fst_inc_wts_;
float global_beam_, lattice_beam_, am_scale_;

GrpcEngine::GrpcEngine(grpc::ServerReaderWriter<Response, Request> *stream, std::shared_ptr<FUNASR_HANDLE> asr_handler)
    : stream_(stream),
      asr_handle_(std::move(asr_handler)) {
    request_ = std::make_shared<Request>();
    response_ = std::make_shared<Response>();
}

void GrpcEngine::DecodeThreadFunc() {
    std::string asr_result;
    std::string asr_timestamp;
    try {
        FUNASR_DEC_HANDLE decoder_handle = FunASRWfstDecoderInit(*asr_handle_, ASR_OFFLINE, global_beam_, lattice_beam_, am_scale_);
        while (!is_end_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        LOG(INFO) << "audio_buffer: " << audio_buffer_.size() << " bytes";
        FUNASR_RESULT result = FunOfflineInferBuffer(*asr_handle_, audio_buffer_.c_str(), audio_buffer_.size(), RASR_NONE, nullptr, *hotwords_embedding_, sampling_rate_, encoding_, itn_, decoder_handle, "auto", true);
        if (result != nullptr) {
            asr_result = FunASRGetResult(result, 0);
            asr_timestamp = FunASRGetStamp(result);
            FunASRFreeResult(result);
        }
        FunASRWfstDecoderUninit(decoder_handle);
    } catch (const std::exception &e) {
        LOG(ERROR) << e.what();
    }
    Response response;
    response.set_mode(DecodeMode::offline);
    response.set_text(asr_result);
    response.set_is_final(true);
    response.set_timestamp(asr_timestamp);
    stream_->Write(response);
}

void GrpcEngine::OnSpeechStart() {
    if (request_->sampling_rate() != 0) {
        sampling_rate_ = request_->sampling_rate();
    }
    LOG(INFO) << "sampling_rate: " << sampling_rate_;

    switch (request_->wav_format()) {
    case WavFormat::pcm:
        encoding_ = "pcm";
        break;
    default:
        break;
    }
    LOG(INFO) << "encoding: " << encoding_;

    std::string mode_str;
    switch (request_->mode()) {
    case DecodeMode::offline:
        mode_ = ASR_OFFLINE;
        mode_str = "offline";
        break;
    case DecodeMode::online:
        mode_ = ASR_ONLINE;
        mode_str = "online";
        break;
    case DecodeMode::two_pass:
        mode_ = ASR_TWO_PASS;
        mode_str = "two_pass";
        break;
    default:
        break;
    }
    LOG(INFO) << "mode: " << mode_str;

    itn_ = request_->itn();
    LOG(INFO) << "itn: " << (itn_ ? "true" : "false");

    std::unordered_map<std::string, int> hotword_map;
    std::string nn_hotwords;
    for (const auto &pair: request_->hotwords()) {
        hotword_map.insert(pair);
    }
    hotword_map.insert(hws_map_.begin(), hws_map_.end());
    for (const auto &pair: hotword_map) {
        if (!nn_hotwords.empty()) {
            nn_hotwords += " ";
        }
        nn_hotwords += pair.first;
    }
    hotwords_embedding_ = std::make_shared<std::vector<std::vector<float>>>(CompileHotwordEmbedding(*asr_handle_, nn_hotwords));

    is_start_ = true;
    decode_thread_ = std::make_shared<std::thread>(&GrpcEngine::DecodeThreadFunc, this);
}

void GrpcEngine::OnSpeechData() {
    p_mutex_->lock();
    audio_buffer_ += request_->audio_data();
    p_mutex_->unlock();
}

void GrpcEngine::OnSpeechEnd() {
    is_end_ = true;
    LOG(INFO) << "read all pcm data, wait for decoding thread";
    if (decode_thread_ != nullptr) {
        decode_thread_->join();
    }
}

void GrpcEngine::operator()() {
    try {
        LOG(INFO) << "start engine main loop";
        while (stream_->Read(request_.get())) {
            LOG(INFO) << "receive data";
            if (!is_start_) {
                OnSpeechStart();
            }
            OnSpeechData();
            if (request_->is_final()) {
                break;
            }
        }
        OnSpeechEnd();
        LOG(INFO) << "Recognize: done";
    } catch (std::exception const &e) {
        LOG(ERROR) << e.what();
    }
}

GrpcService::GrpcService(std::map<std::string, std::string> &model_path, int model_thread_num)
    : model_path_(model_path) {
    asr_handler_ = std::make_shared<FUNASR_HANDLE>(FunOfflineInit(model_path_, model_thread_num));
    LOG(INFO) << "model loaded successfully";
    // TODO: warmup
}

grpc::Status GrpcService::Recognize(grpc::ServerContext *context, grpc::ServerReaderWriter<Response, Request> *stream) {
    LOG(INFO) << "Recognize: start";
    GrpcEngine engine(stream, asr_handler_);
    std::thread t(std::move(engine));
    t.join();
    return grpc::Status::OK;
}

void GetValue(TCLAP::ValueArg<std::string> &value_arg, const std::string &key, std::map<std::string, std::string> &config) {
    if (value_arg.isSet()) {
        config.insert({key, value_arg.getValue()});
        LOG(INFO) << key << ": " << value_arg.getValue();
    }
}

int main(int argc, char *argv[]) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    TCLAP::CmdLine cmd("funasr-grpc-server", ' ', "0.0.1");
    TCLAP::ValueArg<std::string> model_dir("", MODEL_DIR, "asr offline model path, which contains model.onnx, config.yaml, am.mvn", true, "", "string");
    TCLAP::ValueArg<std::string> quantize("", QUANTIZE, "(default: false) load the model of model.onnx in model_dir. if set true, load the model of model_quant.onnx in model_dir", false, "false", "string");
    TCLAP::ValueArg<std::string> bladedisc("", BLADEDISC, "(default: true) load the model of bladedisc in model_dir.", false, "true", "string");
    TCLAP::ValueArg<std::string> vad_dir("", VAD_DIR, "vad online model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
    TCLAP::ValueArg<std::string> vad_quant("", VAD_QUANT, "(default: false) load the model of model.onnx in vad_dir. if set true, load the model of model_quant.onnx in vad_dir", false, "false", "string");
    TCLAP::ValueArg<std::string> punc_dir("", PUNC_DIR, "punc online model path, which contains model.onnx, punc.yaml", false, "", "string");
    TCLAP::ValueArg<std::string> punc_quant("", PUNC_QUANT, "(default: false) load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir", false, "false", "string");
    TCLAP::ValueArg<std::string> itn_dir("", ITN_DIR, "itn model path, which contains zh_itn_tagger.fst, zh_itn_verbalizer.fst", false, "", "string");
    TCLAP::ValueArg<std::string> lm_dir("", LM_DIR, "LM model path, which contains compiled models: TLG.fst, config.yaml", false, "", "string");
    TCLAP::ValueArg<std::string> hotword("", HOTWORD, "hotword file, one hotword per line. format: \"{hotword} {weight}\" (e.g. 阿里巴巴 20)", false, "", "string");
    TCLAP::ValueArg<int> fst_inc_wts("", FST_INC_WTS, "fst hotwords incremental bias", false, 20, "int");
    TCLAP::ValueArg<float> global_beam("", GLOB_BEAM, "decoding beam for beam searching", false, 3.0, "float");
    TCLAP::ValueArg<float> lattice_beam("", LAT_BEAM, "lattice generation beam for beam searching", false, 3.0, "float");
    TCLAP::ValueArg<float> am_scale("", AM_SCALE, "acoustic scale for beam searching", false, 10.0, "float");
    TCLAP::ValueArg<std::string> host("", "host", "host", false, "0.0.0.0", "string");
    TCLAP::ValueArg<int> port("", "port", "port", false, 80, "int");
    TCLAP::ValueArg<int> worker_num("", "worker-num", "number of gRPC workers", false, 4, "int");
    TCLAP::ValueArg<int> model_thread_num("", "model-thread-num", "number of model threads", false, 1, "int");

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(bladedisc);
    cmd.add(vad_dir);
    cmd.add(vad_quant);
    cmd.add(punc_dir);
    cmd.add(punc_quant);
    cmd.add(itn_dir);
    cmd.add(lm_dir);
    cmd.add(hotword);
    cmd.add(fst_inc_wts);
    cmd.add(global_beam);
    cmd.add(lattice_beam);
    cmd.add(am_scale);
    cmd.add(host);
    cmd.add(port);
    cmd.add(worker_num);
    cmd.add(model_thread_num);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(bladedisc, BLADEDISC, model_path);
    GetValue(vad_dir, VAD_DIR, model_path);
    GetValue(vad_quant, VAD_QUANT, model_path);
    GetValue(punc_dir, PUNC_DIR, model_path);
    GetValue(punc_quant, PUNC_QUANT, model_path);
    GetValue(itn_dir, ITN_DIR, model_path);
    GetValue(lm_dir, LM_DIR, model_path);
    GetValue(hotword, HOTWORD, model_path);

    auto hotword_path = hotword.getValue();
    funasr::ExtractHws(hotword_path, hws_map_);
    fst_inc_wts_ = fst_inc_wts.getValue();
    global_beam_ = global_beam.getValue();
    lattice_beam_ = lattice_beam.getValue();
    am_scale_ = am_scale.getValue();

    std::string server_address = host.getValue() + ":" + std::to_string(port.getValue());
    GrpcService service(model_path, model_thread_num.getValue());

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS, worker_num.getValue());
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    LOG(INFO) << "listening on " << server_address;
    server->Wait();
    google::ShutdownGoogleLogging();
    return 0;
}
