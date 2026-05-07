import math

# Implement numpy like array and operations in pure Python to get a feel what we need to do in C later
class ArrayND:
    
    def __init__(self, data: list[float], shape: tuple[int], stride: list[int] = None) -> None:
        self.data: list[float] = data
        self.shape: tuple[int] = shape
        self.n = len(shape)
        self.stride: list[int] = [0] * self.n
        self.offset: int = 0
        
        if stride is None:
            self.stride[self.n-1] = 1
            for i in range(self.n-2, -1, -1):
                self.stride[i] = self.stride[i+1] * self.shape[i+1]


        
    def __getitem__(self, indices: tuple[int] | int) -> float:
        # for 1D array convert int to tuple[int]
        if isinstance(indices, int):
            indices = (indices, )

        if len(indices) != self.n:
            raise IndexError(f"Očekivano {self.n} dimenzija, dobijeno {len(indices)}")
        
        position: int = self.offset
        for i, idx in enumerate(indices):
            position += self.stride[i] * idx

        return self.data[position]
    
    def reshape(self, *new_shape: int | tuple) -> ArrayND:
        """
        Changes the shape of ArrayNd. The number of elements must be retained.
        """
        # allow both reshape(2, 3) and reshape((2, 3))
        if len(new_shape) == 1 and isinstance(new_shape[0], tuple):
            new_shape = new_shape[0]

        old_numel: int = math.prod(self.shape)
        new_numel: int = math.prod(new_shape) if new_shape else 0

        if old_numel != new_numel:
            raise ValueError(f"Impossible to reshape ArrayNd of shape: {self.shape} to shape: {new_shape}, absolute size cannot change!")
        
        return ArrayND(self.data, new_shape)
    

    def transpose(self, *axis) -> ArrayND:
        """
        Generates a new ArrayND whose dimensions are the permutation of the original
        """
        axis: tuple[int] = ()

        if not axis:
            # if there are no arguments reverse the dimensions (default e.g. for Matrix rows become columns and columns become rows)
            axis = tuple(reversed(range(self.n)))
        elif len(axis) == 1 and isinstance(axis[0], tuple):
            axis = axis[0]

        if len(axis) != self.n:
            raise ValueError(f"Expected {self.n} axis, gotten {len(axis)}")

        new_shape: tuple[int] = tuple(self.shape[i] for i in axis)
        new_stride: list[int] = list(self.stride[i] for i in axis)

        return ArrayND(self.data, new_shape, new_stride)

    @property
    def T(self) -> 'ArrayND':
        return self.transpose()

# Sta moram da imam

"""
Nekakav linearni storage
velicine po dimenzijama
offset
stride - torka ciji i-ti element kaze za koliko elemenata treba da se pomerim po fizickoj reprezentaciji NDarray-a da bi se pomerio na sledeci po ovoj dimenziji

ako imam mat[100][100], ja je cuvam linearizovano u data kao niz od 10 000 elemenata
size = (100, 100), a stride = (100, 1), jer sam fizicku reprezentaciju napravio tako sto sam linearizovao po vrstama/redovima
pa da bih presao na sledeci element po dimenziji 2 dovoljno je da predjem na sledeci element u fizickoj reprezentaciji, jer sam tako i pakovao, tj u sledecu kolonu
prelazim pomeranjem, jedan, ali da bi se pomerio za 1 red, tj. u sledeci red treba da se pomerim za 100 mesta, tj za size[i-1]
ako bih imao ovakvih 5 matrica, tj 3D niz size = (5, 100, 100), ja imam reprezentaciju koja je napravljena pakovanjem 5 uzastopnih matrica iz prethodnog primera,
upravo na isti nacin => stride[2] = 1, stride[1] = 100, a stride[0] kaze koliko treba elemenata da se pomerim u vizickoj reprezentaciji da bih presao u sledeci u ovoj
dimenziji, ova dimenzija predstavlja niz matrica, tako da treba da predjem na sledecu matricu, a matrica ima 100x100 elemenata => stride[3] = 10 000
Zakljucak stecen induktivno: stride[i] = prod(size[j]) for j = i + 1 to N
1D niz je niz skalara
2D niz je niz nizova skalara = matrica
3D niz je niz nizova nizova skalara = niz matrica
4D niz je niz nizova matrica = matrica matrica
pomeranje po nekoj dimenziji za 1 je zapravo kako doci do sledeceg elementa u tom nizu, bilo da je to skalara(za 1D), niz(za matricu), matrica(za 3D) ili sta god
Odgovor: pomericemo se na sledeci element niza samo kada predjemo sve elemente kolekcije koja cini jedan njegov element i ovaj iskaz je rekurzivan

Samo je ovo sto dalje pisem sve reverse jer sam 0 nazvao element skroz desno => zakljucak je sledeci: stride[n-1] = 1; stride[i] = size[i+1]*stride[i+1]
ako imamo size[n] - mislim na celokupan niz, cemu je jednak stride[n]
        Bazni slucaj: stride[0] = 1
        Indukcijska hipoteza: stride[n] = Prod(size[j]) for j = 0 to n - 1
        Dokazati da hipoteza => stride[n+1] = Prod(size[j]) for j = 0 to n
            neka imamo niz a koji ima n+1 dimenziju, tj on je niz n dimenzionih nizova
            njegova linearizovana reprezentacija je uzastopno redjanje n dimenzionih nizova po njihovoj linearizovanoj reprezentaciji
            stride[n+1] kaze koliko je potrebno da se pomerimo elemenata po linearizovanoj reprezentaciji ndimenzionog niza da bismo presli na sledeci element
            niza dimenzije n+1 - Odgovor preskociti njegov element koji ima dimenziju n+1
            Kako to uraditi, tako sto znamo kako u njemu da se pomerimo za jedan element - po hipotezi to je stride[n]
            mi treba da napravimo size[n] puta stride[n] i pomericemo se za jedan u nizu dimenzije n+1
            => stride[n+1] = stride[n] * size[n] = size[n] * Prod(size[j]) for j = 0 to n - 1 = Prod(size[j]) for j = 0 to n => DOKAZANO


Sta hocu da implementiram:
    vektorski proizvod
    standardne aritmeticke operaciije: +, -, *, /, log, exp
    sum, mean
    reshape, view transpose
    broadcasting


    123
    456

    14
    25
    36

    123456
"""