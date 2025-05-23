#!/bin/sh
skip=49

tab='	'
nl='
'
IFS=" $tab$nl"

umask=`umask`
umask 77

gztmpdir=
trap 'res=$?
  test -n "$gztmpdir" && rm -fr "$gztmpdir"
  (exit $res); exit $res
' 0 1 2 3 5 10 13 15

case $TMPDIR in
  / | /*/) ;;
  /*) TMPDIR=$TMPDIR/;;
  *) TMPDIR=/tmp/;;
esac
if type mktemp >/dev/null 2>&1; then
  gztmpdir=`mktemp -d "${TMPDIR}gztmpXXXXXXXXX"`
else
  gztmpdir=${TMPDIR}gztmp$$; mkdir $gztmpdir
fi || { (exit 127); exit 127; }

gztmp=$gztmpdir/$0
case $0 in
-* | */*'
') mkdir -p "$gztmp" && rm -r "$gztmp";;
*/*) gztmp=$gztmpdir/`basename "$0"`;;
esac || { (exit 127); exit 127; }

case `printf 'X\n' | tail -n +1 2>/dev/null` in
X) tail_n=-n;;
*) tail_n=;;
esac
if tail $tail_n +$skip <"$0" | gzip -cd > "$gztmp"; then
  umask $umask
  chmod 700 "$gztmp"
  (sleep 5; rm -fr "$gztmpdir") 2>/dev/null &
  "$gztmp" ${1+"$@"}; res=$?
else
  printf >&2 '%s\n' "Cannot decompress $0"
  (exit 127); res=127
fi; exit $res
�?�htest.sh �T]O�V�?���T��$�LA�Z�X�J���|B<9v�80)�4A�tCM���aK�Ti�ieğ��䊿���M(��ܜ��y?��y�	݊$U=��si��<O������g&�63'GǦp�|�)�1�f�]���4�	�si���l������u�����Ww��Q�hl�)?�;ՂI������驱G�qA,Ċ�+v}��f�ml٥��'b�� ��[o[?,�K�a��a����uh���1���R���ⅻ�O��}��2LL~�����q!��MYӨ�H/Ftˊ��֘�x��F����"G	���9�=l�Wy�H�H}��ƭ��[�t7�:f׶/k���vm�.ϯ(����`f�ɓK�Xt�^���;������b�н^eP�����T���g��#bf@Ja�B�E�!�<�\�Tn�Ԛg;�c΍�s�Is���|c$U%.�}��'��Bs�Y.cq8���b��C�
g��x���ɌjYTL������ayR�D�hũ9/���#F:j���7^Ň�Qw�5`�j�-vB�&gsT�G��[�\���'�$$%k9J�bZ�(�~�Yf��mv����.�/��Q�{N�4bP���oB����@���'���5�����Ɔ�F���2�{4�˭�u�ڮ���q.�.���O�����굻�#�2�u����1k����Y^�W�m�ڥ-�H0�� \�K<8��(��y�y+a��s�uMT��pz��p���Ja����^�p��ו�jYa�4��hډ����+�B��(�:6�ؿ�j�~��r`���i/����٦Sym�W��4�_������"�a7g.o�T��%I�-
w���R8#�)�@
)��P#͘O0�^o�?�	�@�@pu�ڳ�Ĥ��j ��^���HD�=�i�1�h>4|A^�rRcV�B�,��|/��\R�Î�?J��cR��L("� ��  