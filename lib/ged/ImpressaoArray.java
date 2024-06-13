package lib.ged;

class ImpressaoArray {
	
	public ImpressaoArray() {}

	public void printArray(Object array, String nome) {
		String espacamento = "  ";

		if (nome.isEmpty()) {
			System.out.println("Array = [");
		
		} else {
			System.out.println(nome + " = [");
		}

		if (array instanceof int[]) {
			int[] arr = (int[]) array;

			System.out.print(espacamento + arr[0]);
			for (int i = 1; i < arr.length; i++) {
				System.out.print(", " + arr[i]);
			}
			System.out.println();
	
			System.out.println("]");
		
		}else if (array instanceof float[]) {
			float[] arr = (float[]) array;

			System.out.print(espacamento + arr[0]);
			for (int i = 1; i < arr.length; i++) {
				System.out.print(", " + arr[i]);
			}
			System.out.println();
	
			System.out.println("]");
		
		}else if (array instanceof double[]) {
			double[] arr = (double[]) array;

			System.out.print(espacamento + arr[0]);
			for (int i = 1; i < arr.length; i++) {
				System.out.print(", " + arr[i]);
			}
			System.out.println();
	
			System.out.println("]");
		
		}else if (array instanceof String[]) {
			String[] arr = (String[]) array;
			
			System.out.print(espacamento + arr[0]);
			for (int i = 1; i < arr.length; i++) {
				System.out.print(", " + arr[i]);
			}
			System.out.println();
	
			System.out.println("]");
		
		} else {
			throw new IllegalArgumentException("Tipo de array não suportado.");
		}
	}

	public void imprimirArray(Object array) {
		printArray(array, "");
	}

}
