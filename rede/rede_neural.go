package rede

import (
	"math"

	"github.com/glaubergoncalves/go-rede-neural/matriz"
)

type RedeNeural struct {
	Entrada               int
	Escondida             int
	Saida                 int
	BiasEntradaParaOculta matriz.Matriz
	BiasOcultaParaSaida   matriz.Matriz
	PesoEntradaEscondida  matriz.Matriz
	PesoEscondidaSaida    matriz.Matriz
	TaxaAprendizado       float64
}

func sigmoidDerivada(x float64) float64 {
	return x * (1 - x)
}

func (r *RedeNeural) Inicia(entrada, escondida, saida int) {

	r.Entrada = entrada
	r.Escondida = escondida
	r.Saida = saida

	r.BiasEntradaParaOculta = matriz.Matriz{}
	r.BiasOcultaParaSaida = matriz.Matriz{}
	r.PesoEntradaEscondida = matriz.Matriz{}
	r.PesoEntradaEscondida = matriz.Matriz{}

	r.BiasEntradaParaOculta.Iniciar(escondida, 1)
	r.BiasEntradaParaOculta.Randomizar()
	r.BiasOcultaParaSaida.Iniciar(saida, 1)
	r.BiasOcultaParaSaida.Randomizar()

	r.PesoEntradaEscondida.Iniciar(escondida, entrada)
	r.PesoEntradaEscondida.Randomizar()
	r.PesoEscondidaSaida.Iniciar(saida, escondida)
	r.PesoEscondidaSaida.Randomizar()

	r.TaxaAprendizado = 0.1
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Ativacao(matriz matriz.Matriz) {
	for i := 0; i < matriz.Linhas; i++ {
		for j := 0; j < matriz.Colunas; j++ {
			matriz.Dados[i][j] = sigmoid(matriz.Dados[i][j])
		}
	}
}

func (r *RedeNeural) feedForward(_entrada []float64) matriz.Matriz {

	// camada de entrapa para a escondida
	entrada := matriz.ArrayParaMatriz(_entrada)
	escondida := matriz.Multiplicacao(r.PesoEntradaEscondida, entrada)
	escondida = matriz.Soma(escondida, r.BiasEntradaParaOculta)
	Ativacao(escondida)

	// camada de escondida para saida

	saida := matriz.Multiplicacao(r.PesoEscondidaSaida, escondida)
	saida = matriz.Soma(saida, r.BiasOcultaParaSaida)

	Ativacao(saida)
	return saida
}

func (r *RedeNeural) Treinar(arrayEntrada []float64, alvo []float64) {

	// camada de entrapa para a escondida
	entrada := matriz.ArrayParaMatriz(arrayEntrada)
	escondida := matriz.Multiplicacao(r.PesoEntradaEscondida, entrada)
	escondida = matriz.Soma(escondida, r.BiasEntradaParaOculta)
	Ativacao(escondida)

	// camada de escondida para saida

	saida := matriz.Multiplicacao(r.PesoEscondidaSaida, escondida)
	saida = matriz.Soma(saida, r.BiasOcultaParaSaida)
	Ativacao(saida)

	// backpropagation

	// saida --> oculta
	valorEsperado := matriz.ArrayParaMatriz(alvo)
	erroSaida := matriz.Subtracao(valorEsperado, saida)
	derivadaSaida := matriz.Mapeia(saida, func(saida matriz.Matriz, linha, coluna int) float64 {
		return sigmoidDerivada(saida.Dados[linha][coluna])
	})
	escondidaTransposta := escondida.MatrizTransposta()

	gradiente := matriz.Hadamard(derivadaSaida, erroSaida)
	gradiente = matriz.MultiplicacaoEscalar(gradiente, r.TaxaAprendizado)

	r.BiasOcultaParaSaida = matriz.Soma(r.BiasOcultaParaSaida, gradiente)
	pesoEscondidaSaidaDelta := matriz.Multiplicacao(gradiente, escondidaTransposta)
	r.PesoEscondidaSaida = matriz.Soma(r.PesoEscondidaSaida, pesoEscondidaSaidaDelta)

	pesoEscondidaSaidaTransposto := r.PesoEscondidaSaida.MatrizTransposta()
	erroEscondida := matriz.Multiplicacao(pesoEscondidaSaidaTransposto, erroSaida)
	derivadaEscondida := matriz.Mapeia(escondida, func(escondida matriz.Matriz, linha, coluna int) float64 {
		return sigmoidDerivada(escondida.Dados[linha][coluna])
	})
	entradaTransposta := entrada.MatrizTransposta()
	gradienteEscondida := matriz.Hadamard(derivadaEscondida, erroEscondida)
	gradienteEscondida = matriz.MultiplicacaoEscalar(gradienteEscondida, r.TaxaAprendizado)

	// Adjust Bias O->H
	r.BiasEntradaParaOculta = matriz.Soma(r.BiasEntradaParaOculta, gradienteEscondida)

	pesoEntradaEscondidaDelta := matriz.Multiplicacao(gradienteEscondida, entradaTransposta)
	r.PesoEntradaEscondida = matriz.Soma(r.PesoEntradaEscondida, pesoEntradaEscondidaDelta)

}

func (r *RedeNeural) Prever(arrayEntrada []float64) [][]float64 {
	// entrada -> escondida
	entrada := matriz.ArrayParaMatriz(arrayEntrada)

	escondida := matriz.Multiplicacao(r.PesoEntradaEscondida, entrada)
	escondida = matriz.Soma(escondida, r.BiasEntradaParaOculta)

	escondida = matriz.Mapeia(escondida, func(saida matriz.Matriz, linha, coluna int) float64 {
		return sigmoid(escondida.Dados[linha][coluna])
	})

	// escondida -> saida
	saida := matriz.Multiplicacao(r.PesoEscondidaSaida, escondida)
	saida = matriz.Soma(saida, r.BiasOcultaParaSaida)
	saida = matriz.Mapeia(saida, func(saida matriz.Matriz, linha, coluna int) float64 {
		return sigmoid(saida.Dados[linha][coluna])
	})
	arraySaida := saida.Dados

	return arraySaida
}
