package com.stargazer.oauth2test.adapter.service.chat

import com.stargazer.oauth2test.common.CommonZoneDateTime
import com.stargazer.oauth2test.domain.model.chat.ChatLastReadModel
import com.stargazer.oauth2test.domain.model.chat.ChatModel
import com.stargazer.oauth2test.domain.model.chat.ChatRoomModel
import com.stargazer.oauth2test.domain.model.chat.ChatRoomProfileModel
import com.stargazer.oauth2test.port.out.repository.chat.ChatLastReadRepository
import com.stargazer.oauth2test.port.out.repository.chat.ChatRepository
import com.stargazer.oauth2test.port.out.repository.chat.ChatRoomProfileRepository
import com.stargazer.oauth2test.port.out.repository.chat.ChatRoomRepository
import com.stargazer.oauth2test.usecase.chat.ChatUseCase
import com.stargazer.oauth2test.usecase.chat.CheckChatRoomDto
import com.stargazer.oauth2test.usecase.chat.LastReadDto
import org.springframework.stereotype.Service

@Service
class ChatService(
    private val chatRepository: ChatRepository,
    private val roomRepository: ChatRoomRepository,
    private val lastReadRepository: ChatLastReadRepository,
    private val profileRepository: ChatRoomProfileRepository
) : ChatUseCase {
    override fun send(model: ChatModel): ChatModel {
        // chat 저장
        chatRepository.save(model)
        updateOrCreate(model)
        return model
    }

    private fun updateOrCreate(model: ChatModel) {
        // chatRoom 업데이트 (마지막 채팅, 마지막 채팅 보낸시간)
        val findRoom = roomRepository.getById(model.roomId)
        if (findRoom == null) {
            // chatRoom 이 존재하지 않을경우 chatRoomProfile 에도 없을테니 추가 해주기
            // chatRoom list 가 존재하지 않을경우 추가 chatRoomProfile 에 추가 , admin 쪽도 추가
            val senderProfile = profileRepository.getById(model.sender) ?: ChatRoomProfileModel(
                id = model.sender,
                roomIds = mutableListOf()
            )
            if (!senderProfile.roomIds.contains(model.roomId)) {
                senderProfile.roomIds.add(model.roomId)
                profileRepository.upsert(senderProfile)
            }
            val adminProfile = profileRepository.getById("admin") ?: ChatRoomProfileModel(
                id = "admin",
                roomIds = mutableListOf()
            )
            if (!adminProfile.roomIds.contains(model.roomId)) {
                adminProfile.roomIds.add(model.roomId)
                profileRepository.upsert(adminProfile)
            }
        }
        val findRoomModel = findRoom ?: ChatRoomModel(
            id = model.roomId,
            participants = mutableListOf("admin", model.sender),
            lastChat = model.content,
            lastChatTimestamp = model.creationDateTime,
            lastChatId = model.id
        )

        val changeRoom = findRoomModel.copy(
            lastChat = model.content.ifBlank { findRoomModel.lastChat },
            lastChatId = if (model.content.isBlank()) findRoomModel.lastChatId else model.id,
            lastChatTimestamp = if (model.content.isBlank()) model.creationDateTime else findRoomModel.lastChatTimestamp
        )

        if (!changeRoom.participants.contains(model.sender)) changeRoom.participants.add(model.sender)
        roomRepository.upsert(changeRoom)

        // lastRead 업데이트
        val lastReadModel =
            lastReadRepository.getByUserIdAndRoomId(userId = model.sender, roomId = model.roomId) ?: ChatLastReadModel(
                roomId = changeRoom.id,
                userId = model.sender,
                lastChatId = changeRoom.lastChatId,
                lastReadTime = CommonZoneDateTime.now()
            )
        lastReadRepository.upsert(
            lastReadModel.copy(
                lastChatId = changeRoom.lastChatId,
                lastReadTime = CommonZoneDateTime.now()
            )
        )
    }

    override fun getAllChatByRoomId(roomId: String, limit: Int): List<ChatModel> {
        return chatRepository.findAllByChatRoomId(roomId, limit)
    }

    override fun readChat(userId: String, chatRoomId: String, lastChatId: String): CheckChatRoomDto? {
        updateOrCreate(ChatModel(roomId = chatRoomId, sender = userId, content = ""))
        val room = roomRepository.getById(roomId = chatRoomId) ?: return null
        val readDtoList = mutableListOf<LastReadDto>()
        room.participants.forEach {
            lastReadRepository.getByUserIdAndRoomId(userId = it, roomId = chatRoomId)?.let { lastRead ->
                readDtoList.add(
                    LastReadDto(
                        userId = lastRead.userId,
                        lastReadTime = lastRead.lastReadTime,
                        lastReadChatId = lastRead.lastChatId
                    )
                )
            }
        }
        return CheckChatRoomDto(
            lastReadList = readDtoList,
            lastChatShort = room.lastChat,
            lastChatId = room.lastChatId,
            lastChatTime = room.lastChatTimestamp
        )
    }


    override fun getPrevChatList(chatRoomId: String, prevChatId: String, limit: Int): List<ChatModel> {
        return chatRepository.getPrevListByChatId(chatRoomId, prevChatId, limit)
    }

}

