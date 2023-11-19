package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public abstract class Ativacao{
   public void calcular(CamadaDensa camada){
      throw new java.lang.UnsupportedOperationException("Implementar ativação.");
   }

   public void derivada(CamadaDensa camada){
      throw new java.lang.UnsupportedOperationException("Implementar ativação derivada.");
   }
}
