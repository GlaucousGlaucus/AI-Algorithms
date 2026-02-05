start :-
    write('X= '), read(X),
    write('Y= '), read(Y),
    arith(X, Y).

arith(X, Y) :-
    add(X, Y),
    subt(X, Y),
    mult(X, Y),
    div(X, Y).

add(X, Y) :-
    R is X + Y,
    write('R is '), write(R), nl.

subt(X, Y) :-
    R is X - Y,
    write('R is '), write(R), nl.

mult(X, Y) :-
    R is X * Y,
    write('R is '), write(R), nl.

div(X, Y) :-
    R is X / Y,
    write('R is '), write(R), nl.
