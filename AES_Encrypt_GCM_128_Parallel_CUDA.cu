/**
 * @file AES_Encryption_GCM_Mode.cu
 * @brief AES encryption 128 Sequence code evaluation.
 *
 * 
 * 
 *
 * @Author: Madhusudhan Swargam <mswargam@oakland.edu>
 *          Prathiksha Chikkamadal Manjunatha <pchikkamadalman@oakland.edu>
 * @date 2025-Nov
 * @copyright
 */





#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                     \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)


 /*
  *
  * TYPE DEFINITIONS
  *
  */

typedef uint8_t byte;


typedef byte block_t[16];

 /*
 * The state, represented as a two-dimensional array of 16 bytes
 * with rows and columns indexed from 0 to 3. 
 */
typedef byte state_t[4][4];

#define AES_BLOCK_SIZE	16 // 128 bits - 8 bytes
#define AES_KEY_SIZE	16 // 126 bit AES Key
#define AES_NO_ROUNDS	10 // 10 rounds for 128 bit key
  
 /* 
  *
  * Hardcoded values for Project demonstration
  * [Keylen = 128]
  * [IVlen = 96]
  * [PTlen = 128]
  * [AADlen = 160]
  * [Taglen = 112]
  * 
  * Count = 0
  * Key = 87f96a86404a2c793b26d7e12c5aaffa
  * IV = 5c6699381a9360ec83dd98dc
  * PT = 43b2b8c81cfcc1e5a27b171e80dcf74f
  * AAD = f89016b26cea39ea38a038a0f18af53f72f7fd17
  * CT = 4f3112a81a3531261ce900d92b43faf2
  * Tag = c3a2481fc31a33b46c6b64041d5d
  * Source : https://csrc.nist.gov/Projects/cryptographic-algorithm-validation-program/cavp-testing-block-cipher-modes
  */
 
 
 #define AAD_LEN_BITS 	(160)
 #define AAD_LEN_BYTE 	(AAD_LEN_BITS/8)
 #define KEY_LEN_BITS  	(128)
 #define KEY_LEN_BYTE 	(KEY_LEN_BITS/8)
 #define IV_LEN_BITS	(96)
 #define IV_LEN_BYTE	(IV_LEN_BITS/8)
 #define TAG_LEN		(14)
 byte HardCoded_Key[KEY_LEN_BYTE] = {0x87, 0xf9,0x6a, 0x86,
									 0x40, 0x4a, 0x2c, 0x79,
									 0x3b, 0x26, 0xd7, 0xe1,
									 0x2c, 0x5a, 0xaf,0xfa}; // 16 bytes - 128bits 
 byte HardCoded_IV[IV_LEN_BYTE] = {0x5c, 0x66, 0x99, 0x38,
									0x1a, 0x93, 0x60, 0xec,
									0x83, 0xdd, 0x98, 0xdc}; // 12 bytes - 96 bits

 byte HardCoded_AAD[AAD_LEN_BYTE] = {0xf8, 0x90, 0x16, 0xb2,
									 0x6c, 0xea, 0x39, 0xea,
									 0x38, 0xa0, 0x38, 0xa0,
									 0xf1, 0x8a, 0xf5, 0x3f,
									 0x72, 0xf7, 0xfd, 0x17};//20 bytes - 160 bytes

 
 /*
  * SUBBYTES uses substitution table called as S-box.
  *
  * Source : https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf Table 4
  */
 static const byte sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};


//TODO used reference as https://web.mit.edu/freebsd/head/contrib/wpa/src/crypto/aes-gcm.c


 /*
  * KEYEXPANSION  invokes 10 fxed words denoted by Rcon Round Constants.
  *
  * Source : https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf Table 5
  */
static const byte Rcon[11] = {
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

/*
 *	Algorithm Parameters
 *
 *
 */
#define Nr		10	//	The number of rounds.
#define Nk 		4	//	The number of 32-bit words comprising the key
#define Nb 		4	//	The number of columns comprising the state


/*
 * 5.1.1 SUBBYTES() is an invertible, non-linear transformation of 
 * the state in which a substitution table, called an S-box,
 * is applied independently to each byte in the state
 *
 *
 * Source : https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf Chapter 5.1.1
 */
void SubBytes(state_t* state) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            (*state)[j][i] = sbox[(*state)[j][i]];
        }
    }
}

/*
 * 5.1.4 ADDROUNDKEY is a transformation of the state in which a round key 
 * is combined with the state by applying the bitwise XOR operation
 *
 *
 * Source : https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf Chapter 5.1.4
 */
 
 void AddRoundKey(byte round, state_t* state, const byte* RoundKey) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            (*state)[j][i] ^= RoundKey[round * Nb * 4 + i * Nb + j];
        }
    }
}
/*
 * TODO
 *
 * Source : https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf TODO
 */
void ShiftRows(state_t* state) {
    byte temp;
    temp = (*state)[1][0];
    (*state)[1][0] = (*state)[1][1];
    (*state)[1][1] = (*state)[1][2];
    (*state)[1][2] = (*state)[1][3];
    (*state)[1][3] = temp;
    temp = (*state)[2][0];
    (*state)[2][0] = (*state)[2][2];
    (*state)[2][2] = temp;
    temp = (*state)[2][1];
    (*state)[2][1] = (*state)[2][3];
    (*state)[2][3] = temp;
    temp = (*state)[3][0];
    (*state)[3][0] = (*state)[3][3];
    (*state)[3][3] = (*state)[3][2];
    (*state)[3][2] = (*state)[3][1];
    (*state)[3][1] = temp;
}


/*Multiplication in GF(28) */
byte xtime(byte x) {
    return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}


/*
 * TODO
 *
 * Source : https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf TODO
 */
void MixColumns(state_t* state) {
    byte a, b, c, d;
    for (int i = 0; i < 4; ++i) {
        a = (*state)[0][i];
        b = (*state)[1][i];
        c = (*state)[2][i];
        d = (*state)[3][i];
        (*state)[0][i] = xtime(a) ^ (xtime(b) ^ b) ^ c ^ d;
        (*state)[1][i] = a ^ xtime(b) ^ (xtime(c) ^ c) ^ d;
        (*state)[2][i] = a ^ b ^ xtime(c) ^ (xtime(d) ^ d);
        (*state)[3][i] = (xtime(a) ^ a) ^ b ^ c ^ xtime(d);
    }
}

/*
 * KEYEXPANSION() is a routine applied on key to generate 4 ∗ (Nr +1) words
 * The output of the routine consists of a linear array of words,
 * denoted by w[i], where i is in the range 0 ≤ i < 4 ∗ (Nr +1).
 *
 *
 * Source : https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf Chapter 5.2
 */
 void KeyExpansion(byte RoundKey[176], const byte Key[16]) {
    unsigned i, j, k;
    byte temp[4];
	/*Initialize first 4 words W0 - W3 */
    for (i = 0; i < Nk; ++i) {
        RoundKey[i * 4 + 0] = Key[i * 4 + 0];
        RoundKey[i * 4 + 1] = Key[i * 4 + 1];
        RoundKey[i * 4 + 2] = Key[i * 4 + 2];
        RoundKey[i * 4 + 3] = Key[i * 4 + 3];
    }
	
	/* Need to calculate remaining W4 - W43 */
    for (i = Nk; i < Nb * (Nr + 1); ++i) {
        k = (i - 1) * 4;
		/* Left Shift the Word */
        temp[0] = RoundKey[k + 0];
        temp[1] = RoundKey[k + 1];
        temp[2] = RoundKey[k + 2];
        temp[3] = RoundKey[k + 3];
        if (i % Nk == 0) {
            {
				/* Left Shift the Word */
                byte u = temp[0];
                temp[0] = temp[1];
                temp[1] = temp[2];
                temp[2] = temp[3];
                temp[3] = u;
            }
            {
				/* Substitute the S box word */
                temp[0] = sbox[temp[0]];
                temp[1] = sbox[temp[1]];
                temp[2] = sbox[temp[2]];
                temp[3] = sbox[temp[3]];
            }
            temp[0] = temp[0] ^ Rcon[i / Nk];
        }
		/* Update the key*/
        j = i * 4;
        k = (i - Nk) * 4;
        RoundKey[j + 0] = RoundKey[k + 0] ^ temp[0];
        RoundKey[j + 1] = RoundKey[k + 1] ^ temp[1];
        RoundKey[j + 2] = RoundKey[k + 2] ^ temp[2];
        RoundKey[j + 3] = RoundKey[k + 3] ^ temp[3];
    }
}



/*
 * TODO
 *
 * Source : https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf TODO
 */
void AES_encrypt_block(block_t out, const block_t in, const byte RoundKey[176]) {
    state_t state_buf;
    state_t* s = &state_buf;
    for(int i=0; i<4; ++i) {
        for(int j=0; j<4; ++j) {
            (*s)[j][i] = in[i*4 + j];
        }
    }
    AddRoundKey(0, s, RoundKey);
    for (int round = 1; round < Nr; ++round) {
        SubBytes(s);
        ShiftRows(s);
        MixColumns(s);
        AddRoundKey(round, s, RoundKey);
    }
    SubBytes(s);
    ShiftRows(s);
    AddRoundKey(Nr, s, RoundKey);
    for(int i=0; i<4; ++i) {
        for(int j=0; j<4; ++j) {
            out[i*4 + j] = (*s)[j][i];
        }
    }
}


/*
 * TODO 6.3 Multiplication Operation on Blocks 
The • operation on (pairs of) the 2128 possible blocks corresponds to the multiplication operation
for the binary Galois (finite) field of 2128 elements
 *
 * Source : https://csrc.nist.rip/groups/ST/toolkit/BCM/documents/proposedmodes/gcm/gcm-revised-spec.pdf TODO
 */



// GCM GF(2^128) multiplication  Header TODO
void gcm_gf_mult(const block_t x, const block_t y, block_t z) {
    memset(z, 0, 16);
    block_t v;
    memcpy(v, y, 16);
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 8; ++j) {
            if ((x[i] >> (7 - j)) & 1) {
                for (int k = 0; k < 16; ++k) z[k] ^= v[k];
            }
            int lsb = v[15] & 1;
            for (int k = 15; k > 0; --k) {
                v[k] = (v[k] >> 1) | (v[k - 1] << 7);
            }
            v[0] = v[0] >> 1;
            if (lsb) {
                v[0] ^= 0xE1;
            }
        }
    }
}


//  Header TODO
void gcm_inc_counter(block_t ctr) {
    for (int i = 15; i >= 12; --i) {
        ctr[i]++;
        if (ctr[i] != 0) break;
    }
}

// Header TODO
void xor_blocks(block_t out, const block_t a, const block_t b) {
    for (int i = 0; i < 16; ++i) {
        out[i] = a[i] ^ b[i];
    }
}

// GCTR: AES-CTR mode encryption   Header TODO
void gctr(const byte* RoundKey, const block_t ICB, const byte* in, size_t len_bits, byte* out) {
    block_t ctr_block, enc_ctr_block;
    memcpy(ctr_block, ICB, 16);

    size_t total_bytes = (len_bits + 7) / 8;
    size_t offset_bytes = 0;

    while (offset_bytes < total_bytes) {
        AES_encrypt_block(enc_ctr_block, ctr_block, RoundKey);
        
        size_t remaining_bytes = total_bytes - offset_bytes;
        size_t block_len = (remaining_bytes < 16) ? remaining_bytes : 16;

        for(size_t i=0; i < block_len; ++i) {
            out[offset_bytes + i] = in[offset_bytes + i] ^ enc_ctr_block[i];
        }
        
        gcm_inc_counter(ctr_block);
        offset_bytes += 16;
    }

    size_t last_bits = len_bits % 8;
    if (last_bits > 0) {
        byte mask = (0xFF << (8 - last_bits)) & 0xFF;
        out[total_bytes - 1] &= mask;
    }
}

//   Header TODO
void ghash(const block_t H, const byte* X, size_t len_bits, block_t S) {
    block_t Y;
    memcpy(Y, S, 16);

    block_t temp_Y; // Buffer for gcm_gf_mult output
    
    size_t total_bytes = (len_bits + 7) / 8;
    size_t offset = 0;

    // Process full blocks (16 bytes)
    while (offset + 16 <= total_bytes) {
        if ((len_bits % 8 != 0) && (offset + 16 == total_bytes)) {
            break; 
        }
        xor_blocks(Y, Y, X + offset);
        gcm_gf_mult(Y, H, temp_Y); // Use temp buffer
        memcpy(Y, temp_Y, 16); // Copy result back
        offset += 16;
    }

    // Process last block (which may be partial)
    size_t bytes_remaining = total_bytes - offset;
    if (bytes_remaining > 0) {
        block_t last_block;
        memset(last_block, 0, 16);
        memcpy(last_block, X + offset, bytes_remaining);

        size_t last_bits = len_bits % 8;
        if (last_bits > 0) {
            byte mask = (0xFF << (8 - last_bits)) & 0xFF;
            last_block[bytes_remaining - 1] &= mask;
        }
        
        xor_blocks(Y, Y, last_block);
        gcm_gf_mult(Y, H, temp_Y); // Use temp buffer
        memcpy(Y, temp_Y, 16); // Copy result back
    }
    
    memcpy(S, Y, 16);
}

//  Header TODO
int aes_gcm_128_encrypt(
    const byte Key[16],
    const byte IV[12],
    const byte* PT,
    size_t PTlen_bits,
    const byte* AAD,
    size_t AADlen_bits,
    byte* CT,
    byte* Tag,
    size_t Taglen_bytes
) {
    byte RoundKey[176];
    KeyExpansion(RoundKey, Key);

    block_t H, Z;
    memset(Z, 0, 16);
    AES_encrypt_block(H, Z, RoundKey);

    block_t J0;
    memcpy(J0, IV, 12);
    J0[12] = 0; J0[13] = 0; J0[14] = 0; J0[15] = 1;

    block_t S;
    memset(S, 0, 16);
    ghash(H, AAD, AADlen_bits, S);

    block_t ICB;
    memcpy(ICB, J0, 16);
    gcm_inc_counter(ICB);
    gctr(RoundKey, ICB, PT, PTlen_bits, CT);
    
    ghash(H, CT, PTlen_bits, S);

    block_t len_block;
    memset(len_block, 0, 16);
    uint64_t aad_len_bits_64 = AADlen_bits;
    uint64_t pt_len_bits_64 = PTlen_bits;
    
    for (int i = 0; i < 8; ++i) {
        len_block[i] = (aad_len_bits_64 >> (56 - i * 8)) & 0xFF;
        len_block[i + 8] = (pt_len_bits_64 >> (56 - i * 8)) & 0xFF;
    }
    ghash(H, len_block, 16 * 8, S);

    block_t full_tag;
    gctr(RoundKey, J0, S, 16 * 8, full_tag);
    
    memcpy(Tag, full_tag, Taglen_bytes);

    return 0;
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int PT_Len;
  byte *PT; 
  byte *CT;
  byte *TAG;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  float *vec = (float *)wbImport(wbArg_getInputFile(args, 0), &PT_Len);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  // Convert float WbImport data to uint8 data for calculation
  PT = (uint8_t *)malloc(PT_Len * sizeof(byte));
  for (int i = 0; i < PT_Len; i++) { 
    PT[i] = (uint8_t)vec[i];
  }

#if DEBUG_ENABLE
  printf("\n\n Input file Plain Text PT : \n");
  for(int i =0; i < PT_Len ; i++) {
    printf("[%d] = 0x%x : %f \n", i,PT[i], vec[i]);
  }
  printf("\n\n");
#endif

  wbLog(TRACE, "The input length is ", PT_Len);
  printf(" \n\n The PT length is %d \n", PT_Len);
  CT = (byte *)malloc (PT_Len * sizeof(byte));
  TAG = (byte *)malloc(TAG_LEN * sizeof(byte));


  // Launching Sequential
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching Sequential computation");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Perform Sequential computation here
  aes_gcm_128_encrypt(
                      HardCoded_Key,
                      HardCoded_IV,
                      PT,
                      PT_Len*8,
                      HardCoded_AAD,
                      AAD_LEN_BITS,
                      CT, 
                      TAG, 
                      TAG_LEN
  );
  wbTime_stop(Compute, "Performing CUDA computation");

#if DEBUG_ENABLE
	printf("\n\n Calculated Cipher Data : \n");
	for(int i =0; i < PT_Len ; i++) {
		printf("[%d] = %x \n", i,CT[i]);
	}
	printf("\n\n");
#endif
  
	// Convert Cipher to verify with WB library
	float * CT_float = (float *)malloc(PT_Len * sizeof(float));
	for (int i = 0; i < PT_Len; i++) { 
		CT_float[i] = (float)CT[i];
	}
  
  // Verify correctness
  // -----------------------------------------------------
  wbSolution(args, CT_float, PT_Len);

  // Free the memory
  free(CT);
  free(PT);
  free(TAG);
  free(CT_float);
  
  //Return Success
  return 0;
}