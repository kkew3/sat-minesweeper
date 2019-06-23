import org.apache.commons.math3.util.CombinatoricsUtils;

import java.util.Arrays;
import java.util.Iterator;

public class MakeCNFTable {
    private static final int ULIMIT = 8;
    private static final int LLIMIT = 1;

    private static int[][] make_dnf(int n, int k) {
        Iterator<int[]> iter = CombinatoricsUtils.combinationsIterator(n, n - k);
        long cnkl = CombinatoricsUtils.binomialCoefficient(n, n - k);
        if (cnkl > Integer.MAX_VALUE) {
            return null;
        }
        int cnk = (int) cnkl;
        int[][] neg_indices = new int[cnk][];
        for (int i = 0; i < cnk; ++i) {
            neg_indices[i] = iter.next();
        }
        int[][] lists = new int[cnk][n];
        for (int i = 0; i < cnk; ++i) {
            for (int j = 1; j <= n; ++j) {
                lists[i][j - 1] = j;
            }
        }
        for (int i = 0; i < cnk; ++i) {
            for (int j = 0; j < n - k; ++j) {
                int m = neg_indices[i][j];
                lists[i][m] = -lists[i][m];
            }
        }
        return lists;
    }

    public static void main(String[] args) {
        DNF2CNF converter;
        int[][] dnf;
        int[] clause;
        for (int n = LLIMIT; n <= ULIMIT; ++n) {
            for (int k = 0; k <= n; ++k) {
                if (n >= 6 && k > 1 && k < n - 1) {
                    continue;
                }
                if (n >= 9 && k > 0 && k < n) {
                    continue;
                }
                System.out.println(n + "," + k);
                dnf = make_dnf(n, k);
                converter = new DNF2CNF(dnf, n);
                while (converter.hasNext()) {
                    clause = converter.next();
                    if (clause != null) {
                        for (int e : clause) {
                            System.out.print(e);
                            System.out.print(' ');
                        }
                        System.out.println();
                    }
                }
                System.out.println();
            }
        }

    }
}


class CartesianProductIter implements Iterator<int[]> {
    private int[][] lists;
    private int[] indices;
    private Integer n;
    private boolean echoed;

    CartesianProductIter(int[][] lists) {
        this.lists = lists;
        this.indices = new int[lists.length];
        if (lists.length > 0) {
            this.n = lists[0].length;
        }
        this.echoed = false;
    }

    @Override
    public boolean hasNext() {
        if (this.lists.length == 0) {
            return false;
        }
        return !this.echoed;
    }

    @Override
    public int[] next() {
        int[] e = new int[this.lists.length];
        for (int i = 0; i < this.lists.length; ++i) {
            e[i] = this.lists[i][this.indices[i]];
        }
        this.echoed = true;

        for (int i = this.lists.length - 1; i >= 0; --i) {
            if (this.indices[i] < this.n - 1) {
                this.echoed = false;
                this.indices[i] += 1;
                for (int j = i + 1; j < this.lists.length; ++j) {
                    this.indices[j] = 0;
                }
                break;
            }
        }
        return e;
    }
}

class DNF2CNF implements Iterator<int[]> {
    private CartesianProductIter cpiter;
    private final int[] ZEROS;
    private int[] ibuf;

    DNF2CNF(int[][] dnf, int n) {
        this.cpiter = new CartesianProductIter(dnf);
        this.ZEROS = new int[n];
        this.ibuf = new int[n];
    }

    private boolean existsInversePair(int[] nums) {
        System.arraycopy(this.ZEROS, 0, this.ibuf, 0, this.ZEROS.length);
        for (int e : nums) {
            int ae = Math.abs(e);
            if (this.ibuf[ae - 1] == 0) {
                this.ibuf[ae - 1] = e;
            } else if (this.ibuf[ae - 1] + e == 0) {
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean hasNext() {
        return this.cpiter.hasNext();
    }

    @Override
    public int[] next() {
        int[] l = this.cpiter.next();
        if (existsInversePair(l)) {
            return null;
        }
        return l;
    }
}
