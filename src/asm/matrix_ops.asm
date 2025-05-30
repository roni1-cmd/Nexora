section .text
global matrix_multiply

; matrix_multiply: Multiplies A (m x n) and B (n x p), stores in C (m x p)
; Inputs:
;   rdi: pointer to A (float32, row-major)
;   rsi: pointer to B (float32, row-major)
;   rdx: pointer to C (float32, row-major, output)
;   rcx: m (rows of A)
;   r8: n (cols of A, rows of B)
;   r9: p (cols of B)
matrix_multiply:
    push rbp
    mov rbp, rsp
    push rbx

    mov rax, rcx        ; m
    mov rbx, r8         ; n
    mov r10, r9         ; p

    ; Loop over rows of A (i)
    xor rcx, rcx        ; i = 0
.outer_loop:
    cmp rcx, rax
    jge .done

    ; Loop over cols of B (j)
    xor r9, r9          ; j = 0
.inner_loop:
    cmp r9, r10
    jge .next_row

    ; Compute C[i,j] = sum(A[i,k] * B[k,j]) for k=0 to n-1
    xorps xmm0, xmm0    ; sum = 0
    xor r11, r11        ; k = 0
.sum_loop:
    cmp r11, rbx
    jge .store_result

    ; A[i,k]: A + i*n*4 + k*4
    mov r12, rcx
    imul r12, rbx
    add r12, r11
    shl r12, 2
    movss xmm1, [rdi + r12]

    ; B[k,j]: B + k*p*4 + j*4
    mov r13, r11
    imul r13, r10
    add r13, r9
    shl r13, 2
    movss xmm2, [rsi + r13]

    ; sum += A[i,k] * B[k,j]
    mulss xmm1, xmm2
    addss xmm0, xmm1

    inc r11
    jmp .sum_loop

.store_result:
    ; C[i,j]: C + i*p*4 + j*4
    mov r12, rcx
    imul r12, r10
    add r12, r9
    shl r12, 2
    movss [rdx + r12], xmm0

    inc r9
    jmp .inner_loop

.next_row:
    inc rcx
    jmp .outer_loop

.done:
    pop rbx
    mov rsp, rbp
    pop rbp
    ret
