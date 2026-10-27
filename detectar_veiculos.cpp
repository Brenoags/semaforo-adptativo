#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// ============================================================
// CONFIGURACAO DO PROJETO
// ============================================================

const string VIDEO_ENTRADA = "videos/Video_trasito_para_teste.mp4";
const string VIDEO_SAIDA = "videos/saida_rastreamento.mp4";

const string MODELO_CFG = "modelos/yolov4-tiny.cfg";
const string MODELO_PESOS = "modelos/yolov4-tiny.weights";

// Regioes provisórias: ajuste conforme a imagem do video.
const Rect FLUXO_A(1100, 550, 820, 250); // Via direita
const Rect FLUXO_B(580, 0, 450, 350);   // Via superior

const float CONFIANCA_MINIMA = 0.30f;
const float LIMIAR_NMS = 0.40f;
const int TAMANHO_REDE = 416;

// Rastreamento
const float DISTANCIA_MAXIMA = 160.0f;
const int QUADROS_PERDIDOS_MAX = 15;
const int TAMANHO_HISTORICO = 12;
const float MOVIMENTO_MAXIMO_PARADO = 18.0f;

// Tempos de simulacao acelerada para demonstracao.
// Nao utilizar estes valores diretamente em vias reais.
const float TEMPO_VERDE_TOTAL = 12.0f;
const float TEMPO_VERDE_MINIMO = 2.0f;
const float TEMPO_AMARELO = 1.0f;
const float TEMPO_LIMPEZA = 0.5f;

// ============================================================
// ESTRUTURAS
// ============================================================

struct Deteccao {
    Rect caixa;
    int classe;
    float confianca;
};

struct Veiculo {
    int id = 0;
    int classe = -1;
    Point2f centro;
    vector<Point2f> historico;
    int quadrosPerdidos = 0;
    bool atualizado = false;
};

struct Contagem {
    int motos = 0;
    int carros = 0;
    int onibus = 0;
    int caminhoes = 0;

    int total() const {
        return motos + carros + onibus + caminhoes;
    }

    float demanda() const {
        return motos * 0.7f
             + carros * 1.0f
             + onibus * 2.0f
             + caminhoes * 2.5f;
    }
};

// IDs das classes COCO usadas pelo YOLO.
bool classeVeiculo(int classe) {
    return classe == 2 || classe == 3 ||
           classe == 5 || classe == 7;
}

string nomeClasse(int classe) {
    switch (classe) {
        case 2: return "Carro";
        case 3: return "Moto";
        case 5: return "Onibus";
        case 7: return "Caminhao";
        default: return "Desconhecido";
    }
}

void contarClasse(Contagem& c, int classe) {
    switch (classe) {
        case 2: c.carros++; break;
        case 3: c.motos++; break;
        case 5: c.onibus++; break;
        case 7: c.caminhoes++; break;
    }
}

bool centroDentro(const Rect& roi, Point2f centro) {
    return roi.contains(Point(
        static_cast<int>(centro.x),
        static_cast<int>(centro.y)
    ));
}

void texto(Mat& imagem, const string& mensagem, Point pos,
           Scalar cor = Scalar(255, 255, 255),
           double escala = 0.62) {
    putText(imagem, mensagem, pos, FONT_HERSHEY_SIMPLEX,
            escala, Scalar(0, 0, 0), 4, LINE_AA);
    putText(imagem, mensagem, pos, FONT_HERSHEY_SIMPLEX,
            escala, cor, 2, LINE_AA);
}

// ============================================================
// DETECCAO YOLOV4-TINY
// ============================================================

vector<Deteccao> detectarVeiculos(
    Net& rede,
    const Mat& quadro,
    const vector<String>& nomesSaida
) {
    Mat blob;

    blobFromImage(
        quadro,
        blob,
        1.0 / 255.0,
        Size(TAMANHO_REDE, TAMANHO_REDE),
        Scalar(),
        true,
        false
    );

    rede.setInput(blob);

    vector<Mat> saidas;
    rede.forward(saidas, nomesSaida);

    // DIAGNOSTICO: examina os valores brutos retornados pelo YOLO
    // somente no primeiro quadro, sem alterar a deteccao.
    static bool primeiroDiagnostico = true;

    if (primeiroDiagnostico) {
        float maiorObjeto = 0.0f;
        float maiorProbabilidadeClasse = 0.0f;
        float maiorConfiancaVeiculo = 0.0f;
        int classeMaiorConfianca = -1;
        float menorCoordenada = 1.0f;
        float maiorCoordenada = 0.0f;
        int linhasValidas = 0;

        cout << "\\n=== DIAGNOSTICO DOS VALORES YOLO ===\\n";
        cout << "Numero de saidas: " << saidas.size() << "\\n";

        for (size_t s = 0; s < saidas.size(); s++) {
            const Mat& saida = saidas[s];
            cout << "Saida " << s
                 << " - linhas: " << saida.rows
                 << " - colunas: " << saida.cols
                 << " - dimensoes: " << saida.dims
                 << " - tipo: " << saida.type() << "\\n";

            if (saida.dims != 2 || saida.cols < 6 ||
                saida.type() != CV_32F) {
                cout << "AVISO: formato inesperado nesta saida.\\n";
                continue;
            }

            for (int i = 0; i < saida.rows; i++) {
                const float* dados = saida.ptr<float>(i);
                linhasValidas++;

                for (int k = 0; k < 4; k++) {
                    menorCoordenada = min(menorCoordenada, dados[k]);
                    maiorCoordenada = max(maiorCoordenada, dados[k]);
                }

                maiorObjeto = max(maiorObjeto, dados[4]);

                float maiorProbabilidade = 0.0f;
                int classe = -1;

                for (int c = 5; c < saida.cols; c++) {
                    if (dados[c] > maiorProbabilidade) {
                        maiorProbabilidade = dados[c];
                        classe = c - 5;
                    }
                }

                maiorProbabilidadeClasse =
                    max(maiorProbabilidadeClasse, maiorProbabilidade);

                if (classe >= 0 && classeVeiculo(classe)) {
                    float confianca = dados[4] * maiorProbabilidade;
                    if (confianca > maiorConfiancaVeiculo) {
                        maiorConfiancaVeiculo = confianca;
                        classeMaiorConfianca = classe;
                    }
                }
            }
        }

        cout << "Linhas analisadas: " << linhasValidas << "\\n";
        cout << "Maior valor de objeto: " << maiorObjeto << "\\n";
        cout << "Maior probabilidade de classe: "
             << maiorProbabilidadeClasse << "\\n";
        cout << "Maior confianca combinada de veiculo: "
             << maiorConfiancaVeiculo << "\\n";
        cout << "Classe dessa confianca: "
             << classeMaiorConfianca << " (";
        if (classeMaiorConfianca >= 0)
            cout << nomeClasse(classeMaiorConfianca);
        else
            cout << "nenhuma classe de veiculo";
        cout << ")\\n";
        cout << "Faixa das primeiras 4 coordenadas: "
             << menorCoordenada << " ate " << maiorCoordenada << "\\n";
        cout << "===================================\\n";
        primeiroDiagnostico = false;
    }

    vector<Rect> caixas;
    vector<float> confiancas;
    vector<int> classes;

    // As coordenadas x, y, largura e altura da saida YOLO estao
    // normalizadas entre 0 e 1. Portanto, convertemos diretamente
    // para pixels usando as dimensoes do quadro original.
    float escalaX = static_cast<float>(quadro.cols);
    float escalaY = static_cast<float>(quadro.rows);

    for (const Mat& saida : saidas) {
        // A saida do Darknet geralmente tem uma deteccao por linha.
        for (int i = 0; i < saida.rows; i++) {
            const float* dados = saida.ptr<float>(i);

            if (saida.cols < 6) continue;

            float objeto = dados[4];

            int classeEscolhida = -1;
            float maiorProbabilidade = 0.0f;

            for (int c = 5; c < saida.cols; c++) {
                if (dados[c] > maiorProbabilidade) {
                    maiorProbabilidade = dados[c];
                    classeEscolhida = c - 5;
                }
            }

            if (classeEscolhida < 0 ||
                !classeVeiculo(classeEscolhida)) {
                continue;
            }

            float confianca = objeto * maiorProbabilidade;

            if (confianca < CONFIANCA_MINIMA) continue;

            float cx = dados[0] * escalaX;
            float cy = dados[1] * escalaY;
            float largura = dados[2] * escalaX;
            float altura = dados[3] * escalaY;

            int x = static_cast<int>(cx - largura / 2.0f);
            int y = static_cast<int>(cy - altura / 2.0f);
            int w = static_cast<int>(largura);
            int h = static_cast<int>(altura);

            Rect caixa(x, y, w, h);
            caixa &= Rect(0, 0, quadro.cols, quadro.rows);

            if (caixa.area() <= 0) continue;

            caixas.push_back(caixa);
            confiancas.push_back(confianca);
            classes.push_back(classeEscolhida);
        }
    }

    vector<int> indices;
    NMSBoxes(
        caixas,
        confiancas,
        CONFIANCA_MINIMA,
        LIMIAR_NMS,
        indices
    );

    vector<Deteccao> resultado;

    for (int indice : indices) {
        resultado.push_back({
            caixas[indice],
            classes[indice],
            confiancas[indice]
        });
    }

    return resultado;
}

// ============================================================
// RASTREAMENTO DOS VEICULOS
// ============================================================

void atualizarRastreamento(
    vector<Veiculo>& veiculos,
    const vector<Deteccao>& deteccoes,
    int& proximoId
) {
    for (auto& v : veiculos) {
        v.atualizado = false;
        v.quadrosPerdidos++;
    }

    for (const auto& d : deteccoes) {
        Point2f centro(
            d.caixa.x + d.caixa.width / 2.0f,
            d.caixa.y + d.caixa.height / 2.0f
        );

        int melhor = -1;
        float menorDistancia = DISTANCIA_MAXIMA;

        // Cada deteccao pode atualizar no maximo um veiculo.
        for (int i = 0; i < static_cast<int>(veiculos.size()); i++) {
            const Veiculo& v = veiculos[i];

            if (v.atualizado || v.classe != d.classe) continue;

            float distancia = static_cast<float>(
                norm(centro - v.centro)
            );

            if (distancia < menorDistancia) {
                menorDistancia = distancia;
                melhor = i;
            }
        }

        if (melhor >= 0) {
            Veiculo& v = veiculos[melhor];

            v.centro = centro;
            v.quadrosPerdidos = 0;
            v.atualizado = true;
            v.historico.push_back(centro);

            if (v.historico.size() >
                static_cast<size_t>(TAMANHO_HISTORICO)) {
                v.historico.erase(v.historico.begin());
            }
        } else {
            Veiculo novo;
            novo.id = proximoId++;
            novo.classe = d.classe;
            novo.centro = centro;
            novo.historico.push_back(centro);
            novo.quadrosPerdidos = 0;
            novo.atualizado = true;

            veiculos.push_back(novo);
        }
    }

    veiculos.erase(
        remove_if(
            veiculos.begin(),
            veiculos.end(),
            [](const Veiculo& v) {
                return v.quadrosPerdidos > QUADROS_PERDIDOS_MAX;
            }
        ),
        veiculos.end()
    );
}

// Considera parado quando o centro mudou pouco ao longo
// do historico. A classificacao e uma estimativa visual.
bool estaParado(const Veiculo& v) {
    if (v.historico.size() <
        static_cast<size_t>(TAMANHO_HISTORICO)) {
        return false;
    }

    Point2f inicio = v.historico.front();
    Point2f fim = v.historico.back();

    return norm(fim - inicio) <= MOVIMENTO_MAXIMO_PARADO;
}

// ============================================================
// CONTAGEM E ADAPTACAO DO SEMAFORO
// ============================================================

pair<float, float> calcularVerdes(float demandaA, float demandaB) {
    float soma = demandaA + demandaB;

    if (soma <= 0.0f) {
        return {TEMPO_VERDE_TOTAL / 2.0f,
                TEMPO_VERDE_TOTAL / 2.0f};
    }

    float verdeA = TEMPO_VERDE_TOTAL * demandaA / soma;

    verdeA = max(
        TEMPO_VERDE_MINIMO,
        min(TEMPO_VERDE_TOTAL - TEMPO_VERDE_MINIMO, verdeA)
    );

    float verdeB = TEMPO_VERDE_TOTAL - verdeA;

    return {verdeA, verdeB};
}

string estadoSemaforo(
    float tempoCiclo,
    float verdeA,
    float verdeB,
    bool fluxoA
) {
    float fimVerdeA = verdeA;
    float fimAmareloA = fimVerdeA + TEMPO_AMARELO;
    float inicioVerdeB = fimAmareloA + TEMPO_LIMPEZA;
    float fimVerdeB = inicioVerdeB + verdeB;
    float fimAmareloB = fimVerdeB + TEMPO_AMARELO;

    if (fluxoA) {
        if (tempoCiclo < fimVerdeA) return "VERDE";
        if (tempoCiclo < fimAmareloA) return "AMARELO";
        return "VERMELHO";
    }

    if (tempoCiclo < inicioVerdeB) return "VERMELHO";
    if (tempoCiclo < fimVerdeB) return "VERDE";
    if (tempoCiclo < fimAmareloB) return "AMARELO";

    return "VERMELHO";
}

// ============================================================
// PROGRAMA PRINCIPAL
// ============================================================

int main(int argc, char* argv[]) {
    string entrada = VIDEO_ENTRADA;

    if (argc > 1) {
        entrada = argv[1];
    }

    try {
        Net rede = readNetFromDarknet(MODELO_CFG, MODELO_PESOS);

        if (rede.empty()) {
            cerr << "ERRO: nao foi possivel carregar o modelo YOLO.\n";
            return 1;
        }

        rede.setPreferableBackend(DNN_BACKEND_OPENCV);
        rede.setPreferableTarget(DNN_TARGET_CPU);

        VideoCapture video(entrada);

        if (!video.isOpened()) {
            cerr << "ERRO: nao foi possivel abrir " << entrada << "\n";
            return 1;
        }

        double fps = video.get(CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;

        int largura = static_cast<int>(video.get(CAP_PROP_FRAME_WIDTH));
        int altura = static_cast<int>(video.get(CAP_PROP_FRAME_HEIGHT));
        int totalQuadros = static_cast<int>(video.get(CAP_PROP_FRAME_COUNT));

        cout << fixed << setprecision(2);
        cout << "FPS: " << fps << "\n";
        cout << "Resolucao: " << largura << "x" << altura << "\n";
        cout << "Quadros: " << totalQuadros << "\n";

        VideoWriter gravador(
            VIDEO_SAIDA,
            VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps,
            Size(largura, altura)
        );

        if (!gravador.isOpened()) {
            cerr << "ERRO: nao foi possivel criar o video de saida.\n";
            return 1;
        }

        vector<String> nomesSaida = rede.getUnconnectedOutLayersNames();

        vector<Veiculo> veiculos;
        int proximoId = 1;
        int numeroQuadro = 0;

        float verdeA = TEMPO_VERDE_TOTAL / 2.0f;
        float verdeB = TEMPO_VERDE_TOTAL / 2.0f;

        int ultimaAtualizacao = -1;

        while (true) {
            Mat quadro;

            if (!video.read(quadro)) break;

            numeroQuadro++;

            vector<Deteccao> deteccoes =
                detectarVeiculos(rede, quadro, nomesSaida);

            // DIAGNOSTICO TEMPORARIO: coordenadas das deteccoes YOLO.
            // Imprime um resumo a cada 30 quadros para investigar o fluxo B.
            if (numeroQuadro % 30 == 0) {
                cout << "\n--- Caixas detectadas no quadro "
                     << numeroQuadro << " ---\n";

                for (const auto& d : deteccoes) {
                    Point centro(
                        d.caixa.x + d.caixa.width / 2,
                        d.caixa.y + d.caixa.height / 2
                    );

                    cout << "Classe: " << nomeClasse(d.classe)
                         << " | Confianca: " << d.confianca
                         << " | Caixa: x=" << d.caixa.x
                         << ", y=" << d.caixa.y
                         << ", largura=" << d.caixa.width
                         << ", altura=" << d.caixa.height
                         << " | Centro: (" << centro.x
                         << ", " << centro.y << ")\n";
                }
            }

            atualizarRastreamento(veiculos, deteccoes, proximoId);

            Contagem contagemA;
            Contagem contagemB;

            int rastreados = 0;
            int parados = 0;

            for (const auto& v : veiculos) {
                if (v.quadrosPerdidos > 5) continue;

                rastreados++;

                // Somente veiculos parados entram na fila.
                if (!estaParado(v)) continue;

                if (centroDentro(FLUXO_A, v.centro)) {
                    contarClasse(contagemA, v.classe);
                    parados++;
                } else if (centroDentro(FLUXO_B, v.centro)) {
                    contarClasse(contagemB, v.classe);
                    parados++;
                }
            }

            float tempoVideo = numeroQuadro / fps;
            int segundoAtual = static_cast<int>(tempoVideo);

            // Atualiza a distribuicao dos verdes a cada segundo
            // do video para permitir demonstracao em clipe curto.
            if (segundoAtual != ultimaAtualizacao) {
                auto tempos = calcularVerdes(
                    contagemA.demanda(),
                    contagemB.demanda()
                );

                verdeA = tempos.first;
                verdeB = tempos.second;
                ultimaAtualizacao = segundoAtual;
            }

            float duracaoCiclo = verdeA + TEMPO_AMARELO
                               + TEMPO_LIMPEZA + verdeB
                               + TEMPO_AMARELO;

            float tempoCiclo = fmod(tempoVideo, duracaoCiclo);

            string estadoA = estadoSemaforo(
                tempoCiclo, verdeA, verdeB, true
            );

            string estadoB = estadoSemaforo(
                tempoCiclo, verdeA, verdeB, false
            );

            // Regioes de interesse
            rectangle(quadro, FLUXO_A, Scalar(255, 120, 0), 3);
            rectangle(quadro, FLUXO_B, Scalar(0, 0, 255), 3);

            texto(
                quadro, "FLUXO A - VIA DIREITA",
                Point(FLUXO_A.x + 5, FLUXO_A.y + 25),
                Scalar(255, 180, 0), 0.6
            );

            texto(
                quadro, "FLUXO B - VIA SUPERIOR",
                Point(FLUXO_B.x + 5, FLUXO_B.y + 25),
                Scalar(0, 100, 255), 0.6
            );

            // Caixas de deteccao
            for (const auto& d : deteccoes) {
                rectangle(quadro, d.caixa, Scalar(0, 255, 0), 2);

                string rotulo = nomeClasse(d.classe) + " " +
                    to_string(static_cast<int>(d.confianca * 100)) + "%";

                texto(
                    quadro,
                    rotulo,
                    Point(d.caixa.x, max(18, d.caixa.y - 5)),
                    Scalar(0, 255, 0),
                    0.45
                );
            }

            // Painel de contagem e demanda
            rectangle(
                quadro, Rect(10, 10, 600, 320),
                Scalar(25, 25, 25), FILLED
            );

            texto(quadro, "SEMAFORO ADAPTATIVO",
                  Point(25, 40), Scalar(0, 255, 255), 0.7);

            texto(quadro,
                "A - Carros: " + to_string(contagemA.carros) +
                "  Motos: " + to_string(contagemA.motos),
                Point(25, 75));

            texto(quadro,
                "A - Onibus: " + to_string(contagemA.onibus) +
                "  Caminhoes: " + to_string(contagemA.caminhoes),
                Point(25, 105));

            texto(quadro,
                "Demanda A: " + to_string(contagemA.demanda()),
                Point(25, 135), Scalar(255, 180, 80));

            texto(quadro,
                "B - Carros: " + to_string(contagemB.carros) +
                "  Motos: " + to_string(contagemB.motos),
                Point(25, 175));

            texto(quadro,
                "B - Onibus: " + to_string(contagemB.onibus) +
                "  Caminhoes: " + to_string(contagemB.caminhoes),
                Point(25, 205));

            texto(quadro,
                "Demanda B: " + to_string(contagemB.demanda()),
                Point(25, 235), Scalar(100, 160, 255));

            texto(quadro,
                "Verde A: " + to_string(verdeA) + " s",
                Point(25, 275), Scalar(0, 255, 0));

            texto(quadro,
                "Verde B: " + to_string(verdeB) + " s",
                Point(300, 275), Scalar(0, 255, 0));

            texto(quadro,
                "Deteccoes: " + to_string(deteccoes.size()) +
                " | Rastreados: " + to_string(rastreados) +
                " | Parados: " + to_string(parados),
                Point(25, 310), Scalar(220, 220, 220), 0.48);

            // Indicadores visuais dos semaforos
            auto corEstado = [](const string& estado) {
                if (estado == "VERDE") return Scalar(0, 160, 0);
                if (estado == "AMARELO") return Scalar(0, 180, 220);
                return Scalar(0, 0, 180);
            };

            rectangle(
                quadro, Rect(largura - 210, 20, 190, 75),
                corEstado(estadoA), FILLED
            );
            texto(quadro, "SEMAFORO A",
                  Point(largura - 200, 45));
            texto(quadro, estadoA,
                  Point(largura - 200, 80));

            rectangle(
                quadro, Rect(largura - 210, 110, 190, 75),
                corEstado(estadoB), FILLED
            );
            texto(quadro, "SEMAFORO B",
                  Point(largura - 200, 135));
            texto(quadro, estadoB,
                  Point(largura - 200, 170));

            gravador.write(quadro);

            if (numeroQuadro % 30 == 0) {
                int deteccoesA = 0;
                int deteccoesB = 0;
                for (const auto& d : deteccoes) {
                    Point2f centroD(
                        d.caixa.x + d.caixa.width / 2.0f,
                        d.caixa.y + d.caixa.height / 2.0f
                    );
                    if (centroDentro(FLUXO_A, centroD)) deteccoesA++;
                    if (centroDentro(FLUXO_B, centroD)) deteccoesB++;
                }

                cout << "Quadro " << numeroQuadro
                     << " | Deteccoes YOLO: " << deteccoes.size()
                     << " | Deteccoes na ROI A: " << deteccoesA
                     << " | Deteccoes na ROI B: " << deteccoesB
                     << " | Rastreados: " << rastreados
                     << " | Parados: " << parados
                     << " | Fila A: " << contagemA.total()
                     << " | Demanda A: " << contagemA.demanda()
                     << " | Verde A: " << verdeA << " s"
                     << " | Fila B: " << contagemB.total()
                     << " | Demanda B: " << contagemB.demanda()
                     << " | Verde B: " << verdeB << " s\n";
            }
        }

        video.release();
        gravador.release();

        cout << "\nProcessamento concluido!\n";
        cout << "Video salvo em: " << VIDEO_SAIDA << "\n";

    } catch (const cv::Exception& erro) {
        cerr << "Erro do OpenCV: " << erro.what() << "\n";
        return 1;
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }

    return 0;
}