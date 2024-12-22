package lib.ged;

class OperadorMatrizMultithread {
	
	private int nThreads = 2;

	public OperadorMatrizMultithread(int nThreads) {
		this.nThreads = nThreads;
	}


	public int[][] multiplicarMatrizes(int[][] a, int[][] b) {
		int linA = a.length;
		int colA = a[0].length;
		int colB = b[0].length;
		int linPorThread = linA / nThreads;
		Thread[] threads = new Thread[nThreads];

		int[][] r = new int[linA][colB];

		for (int t = 0; t < nThreads; t++) {
			final int id = t;

			threads[t] = new Thread(() -> {
				int inicio = id * linPorThread;
				int fim = (id == nThreads - 1) ? linA : (id + 1) * linPorThread;

				for (int i = inicio; i < fim; i++) {
					for (int j = 0; j < colB; j++) {
						r[i][j] = 0;
						for (int k = 0; k < colA; k++) {
							r[i][j] += a[i][k] * b[k][j];
						}
					}
				}
			});
			
			threads[t].start();
		}

		try {
			for (Thread thread : threads) {
				thread.join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		return r;
	}


	public float[][] multiplicarMatrizes(float[][] a, float[][] b) {
		int linA = a.length;
		int colA = a[0].length;
		int colB = b[0].length;
		int linPorThread = linA / nThreads;
		Thread[] threads = new Thread[nThreads];

		float[][] r = new float[linA][colB];

		for (int t = 0; t < nThreads; t++) {
			final int id = t;

			threads[t] = new Thread(() -> {
				int inicio = id * linPorThread;
				int fim = (id == nThreads - 1) ? linA : (id + 1) * linPorThread;

				for (int i = inicio; i < fim; i++) {
					for (int j = 0; j < colB; j++) {
						r[i][j] = 0;
						for (int k = 0; k < colA; k++) {
							r[i][j] += a[i][k] * b[k][j];
						}
					}
				}
			});
			
			threads[t].start();
		}

		try {
			for (Thread thread : threads) {
				thread.join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		return r;
	}


	public double[][] multiplicarMatrizes(double[][] a, double[][] b) {
		int linA = a.length;
		int colA = a[0].length;
		int colB = b[0].length;
		int linPorThread = linA / this.nThreads;
		Thread[] threads = new Thread[nThreads];

		double[][] r = new double[linA][colB];

		for (int t = 0; t < this.nThreads; t++) {
			final int id = t;

			threads[t] = new Thread(() -> {
				int inicio = id * linPorThread;
				int fim = (id == this.nThreads - 1) ? linA : (id + 1) * linPorThread;

				for (int i = inicio; i < fim; i++) {
					for (int j = 0; j < colB; j++) {
						r[i][j] = 0;
						for (int k = 0; k < colA; k++) {
							r[i][j] += a[i][k] * b[k][j];
						}
					}
				}
			});
			
			threads[t].start();
		}

		try {
			for (Thread thread : threads) {
				thread.join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		return r;
	}
}
